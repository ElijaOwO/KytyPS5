#include "graphics/shader/recompiler/ir/passes/ResourceWaveAnalysis.h"

#include <cstdint>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

bool ImmediateU32(Value value, uint32_t expected) {
	value = value.Resolve();
	return value.IsImmediate() && value.GetType() == Type::U32 && value.U32() == expected;
}

bool MatchNonZero(Value value, Value mask) {
	value            = value.Resolve();
	mask             = mask.Resolve();
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::INotEqual32 ||
	    inst->NumArgs() != 2u) {
		return false;
	}
	return (inst->Arg(0).Resolve() == mask && ImmediateU32(inst->Arg(1), 0u)) ||
	       (inst->Arg(1).Resolve() == mask && ImmediateU32(inst->Arg(0), 0u));
}

bool MatchFindLsb(Value value, Value mask) {
	value            = value.Resolve();
	mask             = mask.Resolve();
	const auto* inst = value.TryInstruction();
	return inst != nullptr && inst->GetOpcode() == ValueOpcode::FindILsb32 &&
	       inst->NumArgs() == 1u && inst->Arg(0).Resolve() == mask;
}

bool MatchHighLane(Value value, Value high_mask) {
	value                 = value.Resolve();
	high_mask             = high_mask.Resolve();
	const auto* inst      = value.TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::IAdd32 ||
	    inst->NumArgs() != 2u) {
		return false;
	}
	return (MatchFindLsb(inst->Arg(0), high_mask) && ImmediateU32(inst->Arg(1), 32u)) ||
	       (MatchFindLsb(inst->Arg(1), high_mask) && ImmediateU32(inst->Arg(0), 32u));
}

bool MatchBallotComponent(Value value, uint32_t expected_component, Inst*& ballot,
                          Value& condition) {
	value            = value.Resolve();
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::CompositeExtractU32x4 ||
	    inst->NumArgs() != 2u || !ImmediateU32(inst->Arg(1), expected_component)) {
		return false;
	}
	auto* candidate = inst->Arg(0).Resolve().TryInstruction();
	if (candidate == nullptr || candidate->GetOpcode() != ValueOpcode::Ballot ||
	    candidate->NumArgs() != 1u) {
		return false;
	}
	ballot    = candidate;
	condition = candidate->Arg(0).Resolve();
	return true;
}

bool ClearsOnly(Value value, Inst* phi) {
	value            = value.Resolve();
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::BitwiseAnd32 ||
	    inst->NumArgs() != 2u) {
		return false;
	}
	return inst->Arg(0).Resolve().TryInstruction() == phi ||
	       inst->Arg(1).Resolve().TryInstruction() == phi;
}

struct MaskOrigin {
	Inst* ballot = nullptr;
	Value condition;
};

std::optional<MaskOrigin> AnalyzeMaskOrigin(Value value, uint32_t component) {
	value       = value.Resolve();
	auto* phi   = value.TryInstruction();
	if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
	    phi->NumArgs() == 0u || phi->NumArgs() != phi->NumPhiBlocks()) {
		return std::nullopt;
	}

	MaskOrigin origin;
	bool       found_origin = false;
	for (size_t index = 0; index < phi->NumArgs(); index++) {
		auto incoming = phi->Arg(index).Resolve();
		if (ClearsOnly(incoming, phi)) {
			continue;
		}

		Inst* ballot = nullptr;
		Value condition;
		if (!MatchBallotComponent(incoming, component, ballot, condition)) {
			return std::nullopt;
		}
		if (!found_origin) {
			origin       = {.ballot = ballot, .condition = condition};
			found_origin = true;
		} else if (origin.ballot != ballot || origin.condition.Resolve() != condition.Resolve()) {
			return std::nullopt;
		}
	}
	if (!found_origin) {
		return std::nullopt;
	}
	return origin;
}

bool MatchSelector(Value selector, Value& low_mask, Value& high_mask) {
	selector         = selector.Resolve();
	const auto* mask = selector.TryInstruction();
	if (mask != nullptr && mask->GetOpcode() == ValueOpcode::BitwiseAnd32 &&
	    mask->NumArgs() == 2u) {
		if (ImmediateU32(mask->Arg(0), 0x3fu)) {
			selector = mask->Arg(1).Resolve();
		} else if (ImmediateU32(mask->Arg(1), 0x3fu)) {
			selector = mask->Arg(0).Resolve();
		} else {
			return false;
		}
	}

	const auto* choose_low = selector.TryInstruction();
	if (choose_low == nullptr || choose_low->GetOpcode() != ValueOpcode::SelectU32 ||
	    choose_low->NumArgs() != 3u) {
		return false;
	}

	const auto* low_find = choose_low->Arg(1).Resolve().TryInstruction();
	if (low_find == nullptr || low_find->GetOpcode() != ValueOpcode::FindILsb32 ||
	    low_find->NumArgs() != 1u) {
		return false;
	}
	low_mask = low_find->Arg(0).Resolve();
	if (!MatchNonZero(choose_low->Arg(0), low_mask)) {
		return false;
	}

	const auto* choose_high = choose_low->Arg(2).Resolve().TryInstruction();
	if (choose_high == nullptr || choose_high->GetOpcode() != ValueOpcode::SelectU32 ||
	    choose_high->NumArgs() != 3u ||
	    !ImmediateU32(choose_high->Arg(2), 0xffffffffu)) {
		return false;
	}

	const auto* high_add = choose_high->Arg(1).Resolve().TryInstruction();
	if (high_add == nullptr || high_add->GetOpcode() != ValueOpcode::IAdd32 ||
	    high_add->NumArgs() != 2u) {
		return false;
	}
	if (ImmediateU32(high_add->Arg(0), 32u)) {
		const auto* find = high_add->Arg(1).Resolve().TryInstruction();
		if (find == nullptr || find->GetOpcode() != ValueOpcode::FindILsb32 ||
		    find->NumArgs() != 1u) {
			return false;
		}
		high_mask = find->Arg(0).Resolve();
	} else if (ImmediateU32(high_add->Arg(1), 32u)) {
		const auto* find = high_add->Arg(0).Resolve().TryInstruction();
		if (find == nullptr || find->GetOpcode() != ValueOpcode::FindILsb32 ||
		    find->NumArgs() != 1u) {
			return false;
		}
		high_mask = find->Arg(0).Resolve();
	} else {
		return false;
	}
	return MatchNonZero(choose_high->Arg(0), high_mask) &&
	       MatchHighLane(choose_high->Arg(1), high_mask);
}

} // namespace

std::optional<WaterfallReadLaneProof> AnalyzeWaterfallReadLane(
    const Program& program, Value value) {
	(void)program;
	value            = value.Resolve();
	const auto* read = value.TryInstruction();
	if (read == nullptr || read->GetOpcode() != ValueOpcode::ReadLane ||
	    read->NumArgs() != 2u) {
		return std::nullopt;
	}

	Value low_mask;
	Value high_mask;
	if (!MatchSelector(read->Arg(1), low_mask, high_mask)) {
		return std::nullopt;
	}

	const auto low_origin  = AnalyzeMaskOrigin(low_mask, 0u);
	const auto high_origin = AnalyzeMaskOrigin(high_mask, 1u);
	if (!low_origin.has_value() || !high_origin.has_value() ||
	    low_origin->ballot != high_origin->ballot ||
	    low_origin->condition.Resolve() != high_origin->condition.Resolve()) {
		return std::nullopt;
	}

	return WaterfallReadLaneProof {
	    .lane_condition = low_origin->condition.Resolve(),
	    .low_mask = low_mask.Resolve(),
	    .high_mask = high_mask.Resolve(),
	    .may_use_empty_mask_fallback = true,
	};
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
