#include "graphics/shader/recompiler/ir/passes/ResourceAddressAnalysis.h"

#include "graphics/shader/recompiler/ir/passes/ResourceWaveAnalysis.h"

#include <array>
#include <cstdint>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint32_t ImageDescriptorDwords = 8u;
constexpr uint32_t ImageDescriptorStride = ImageDescriptorDwords * sizeof(uint32_t);
constexpr uint32_t MaxAddressImageCandidates = 64u;

bool ImmediateU32(Value value, uint32_t expected) {
	value = value.Resolve();
	return value.IsImmediate() && value.GetType() == Type::U32 && value.U32() == expected;
}

bool ImmediateTrue(Value value) {
	value = value.Resolve();
	return value.IsImmediate() && value.GetType() == Type::U1 && value.U1();
}

Value FindReadLane(Value value, uint32_t depth = 0u) {
	if (depth >= 32u) {
		return {};
	}
	value            = value.Resolve();
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return {};
	}
	if (inst->GetOpcode() == ValueOpcode::ReadLane) {
		return value;
	}

	switch (inst->GetOpcode()) {
		case ValueOpcode::IAdd32:
		case ValueOpcode::UMin32:
		case ValueOpcode::SMin32: {
			Value found;
			for (size_t index = 0; index < inst->NumArgs(); index++) {
				const auto candidate = FindReadLane(inst->Arg(index), depth + 1u);
				if (candidate.IsEmpty()) {
					continue;
				}
				if (!found.IsEmpty() && found.Resolve() != candidate.Resolve()) {
					return {};
				}
				found = candidate;
			}
			return found;
		}
		default: return {};
	}
}

bool MatchDescriptorOffset(Value value, Value& key, uint32_t& stride) {
	value            = value.Resolve();
	const auto* shift = value.TryInstruction();
	uint32_t amount   = 0;
	if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
	    shift->NumArgs() != 2u || !ImmediateU32(shift->Arg(1), 5u)) {
		return false;
	}
	amount = 5u;
	key    = shift->Arg(0).Resolve();
	stride = 1u << amount;
	return true;
}

} // namespace

std::optional<AddressIndirectImageAnalysis> AnalyzeAddressIndirectImage(
    const Program& program, const Inst& handle) {
	if (handle.GetOpcode() != ValueOpcode::GetImageResource ||
	    handle.NumArgs() != ImageDescriptorDwords) {
		return std::nullopt;
	}

	AddressIndirectImageAnalysis result;
	Inst*                        address_handle = nullptr;
	Value                        byte_offset;

	for (uint32_t dword = 0; dword < ImageDescriptorDwords; dword++) {
		auto* read = handle.Arg(dword).Resolve().TryInstruction();
		if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadAddressU32 ||
		    read->NumArgs() != 4u || !ImmediateU32(read->Arg(2), 0u) ||
		    !ImmediateTrue(read->Arg(3))) {
			return std::nullopt;
		}

		const auto flags = read->Flags<MemoryFlags>();
		if (flags.index >= program.memory_info.size()) {
			return std::nullopt;
		}
		const auto& memory = program.memory_info[flags.index];
		if (!IsAddressResourceKind(memory.kind) || memory.data_bits != 32u ||
		    memory.data_dwords != 1u || memory.offset != dword * sizeof(uint32_t)) {
			return std::nullopt;
		}

		auto* current_address = read->Arg(0).Resolve().TryInstruction();
		if (current_address == nullptr ||
		    current_address->GetOpcode() != ValueOpcode::GetAddressResource ||
		    current_address->NumArgs() != 2u ||
		    (address_handle != nullptr && current_address != address_handle)) {
			return std::nullopt;
		}
		address_handle = current_address;

		const auto current_offset = read->Arg(1).Resolve();
		if (dword == 0u) {
			byte_offset = current_offset;
		} else if (!EquivalentValue(program, byte_offset, current_offset)) {
			return std::nullopt;
		}

		result.reads[dword]  = read;
		result.memory[dword] = flags.index;
	}

	Value    key;
	uint32_t stride = 0;
	if (!MatchDescriptorOffset(byte_offset, key, stride) ||
	    stride != ImageDescriptorStride) {
		return std::nullopt;
	}

	const auto read_lane = FindReadLane(key);
	if (read_lane.IsEmpty()) {
		return std::nullopt;
	}
	const auto wave = AnalyzeWaterfallReadLane(program, read_lane);
	if (!wave.has_value()) {
		return std::nullopt;
	}

	const auto range = AnalyzeResourceKeyRange(
	    key, {.condition = wave->lane_condition,
	          .selected_lane_satisfies_condition = true});
	if (!range.has_value() || range->maximum >= MaxAddressImageCandidates) {
		return std::nullopt;
	}

	result.key                       = key;
	result.byte_offset               = byte_offset;
	result.lane_condition            = wave->lane_condition;
	result.address_handle            = address_handle;
	result.conditional_key_range     = *range;
	result.descriptor_stride         = stride;
	result.candidate_count           = range->maximum + 1u;
	result.requires_nonempty_wave_mask = wave->may_use_empty_mask_fallback;
	return result;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
