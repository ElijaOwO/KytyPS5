#include "graphics/shader/recompiler/ir/passes/ResourceWaveAnalysis.h"

#include <cstddef>
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
	Inst* phi = nullptr;
	Value condition;
	Value backedge_value;
	Block* header = nullptr;
	Block* initial_block = nullptr;
	Block* backedge_block = nullptr;
};

std::optional<MaskOrigin> AnalyzeMaskOrigin(Value value, uint32_t component) {
	value       = value.Resolve();
	auto* phi   = value.TryInstruction();
	if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
	    phi->NumArgs() != 2u || phi->NumPhiBlocks() != 2u ||
	    phi->Parent() == nullptr) {
		return std::nullopt;
	}

	MaskOrigin origin;
	origin.phi = phi;
	origin.header = phi->Parent();

	for (size_t index = 0; index < phi->NumArgs(); index++) {
		const auto incoming = phi->Arg(index).Resolve();
		auto* incoming_block = phi->PhiBlock(index);
		if (incoming_block == nullptr) {
			return std::nullopt;
		}
		if (ClearsOnly(incoming, phi)) {
			if (origin.backedge_block != nullptr) {
				return std::nullopt;
			}
			origin.backedge_block = incoming_block;
			origin.backedge_value = incoming;
			continue;
		}

		Inst* ballot = nullptr;
		Value condition;
		if (!MatchBallotComponent(incoming, component, ballot, condition) ||
		    origin.initial_block != nullptr) {
			return std::nullopt;
		}
		origin.ballot        = ballot;
		origin.condition     = condition;
		origin.initial_block = incoming_block;
	}

	if (origin.ballot == nullptr || origin.initial_block == nullptr ||
	    origin.backedge_block == nullptr || origin.backedge_value.IsEmpty() ||
	    origin.initial_block == origin.backedge_block) {
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

std::optional<size_t> BlockIndex(const Program& program, const Block* block) {
	for (size_t index = 0; index < program.blocks.size(); index++) {
		if (program.blocks[index] == block) {
			return index;
		}
	}
	return std::nullopt;
}

bool MatchEitherOrder(Value left, Value right, Value expected_left, Value expected_right) {
	left           = left.Resolve();
	right          = right.Resolve();
	expected_left  = expected_left.Resolve();
	expected_right = expected_right.Resolve();
	return (left == expected_left && right == expected_right) ||
	       (left == expected_right && right == expected_left);
}

bool MatchAnyMaskNonZero(Value condition, Value low, Value high) {
	condition         = condition.Resolve();
	const auto* compare = condition.TryInstruction();
	if (compare == nullptr || compare->GetOpcode() != ValueOpcode::INotEqual32 ||
	    compare->NumArgs() != 2u) {
		return false;
	}

	Value combined;
	if (ImmediateU32(compare->Arg(0), 0u)) {
		combined = compare->Arg(1).Resolve();
	} else if (ImmediateU32(compare->Arg(1), 0u)) {
		combined = compare->Arg(0).Resolve();
	} else {
		return false;
	}

	const auto* bit_or = combined.TryInstruction();
	return bit_or != nullptr && bit_or->GetOpcode() == ValueOpcode::BitwiseOr32 &&
	       bit_or->NumArgs() == 2u &&
	       MatchEitherOrder(bit_or->Arg(0), bit_or->Arg(1), low, high);
}

bool EquivalentLoopCondition(Value value, Value target, const Block* header,
                             const Block* backedge) {
	value  = value.Resolve();
	target = target.Resolve();
	if (value == target) {
		return true;
	}

	const auto* phi = value.TryInstruction();
	if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
	    phi->Parent() != header || phi->NumArgs() != 2u ||
	    phi->NumPhiBlocks() != 2u) {
		return false;
	}

	bool has_target = false;
	bool has_self   = false;
	for (size_t index = 0; index < phi->NumArgs(); index++) {
		const auto incoming = phi->Arg(index).Resolve();
		const auto* block   = phi->PhiBlock(index);
		if (incoming == target) {
			has_target = true;
			continue;
		}
		if (block == backedge && incoming.TryInstruction() == phi) {
			has_self = true;
			continue;
		}
		return false;
	}
	return has_target && has_self;
}

bool ConditionImpliesLoopCondition(Value condition, Value target, const Block* header,
                                   const Block* backedge, uint32_t depth = 0u) {
	if (depth >= 32u) {
		return false;
	}
	condition = condition.Resolve();
	if (EquivalentLoopCondition(condition, target, header, backedge)) {
		return true;
	}
	const auto* inst = condition.TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LogicalAnd ||
	    inst->NumArgs() != 2u) {
		return false;
	}
	return ConditionImpliesLoopCondition(inst->Arg(0), target, header, backedge,
	                                    depth + 1u) ||
	       ConditionImpliesLoopCondition(inst->Arg(1), target, header, backedge,
	                                    depth + 1u);
}

} // namespace

std::optional<WaterfallReadLaneProof> AnalyzeWaterfallReadLane(
    const Program& program, Value value) {
	(void)program;
	value            = value.Resolve();
	const auto* read = value.TryInstruction();
	if (read == nullptr || read->GetOpcode() != ValueOpcode::ReadLane ||
	    read->NumArgs() != 2u || read->Parent() == nullptr) {
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
	    low_origin->condition.Resolve() != high_origin->condition.Resolve() ||
	    low_origin->header != high_origin->header ||
	    low_origin->initial_block != high_origin->initial_block ||
	    low_origin->backedge_block != high_origin->backedge_block) {
		return std::nullopt;
	}

	return WaterfallReadLaneProof {
	    .lane_condition = low_origin->condition.Resolve(),
	    .low_mask = low_mask.Resolve(),
	    .high_mask = high_mask.Resolve(),
	    .low_backedge = low_origin->backedge_value.Resolve(),
	    .high_backedge = high_origin->backedge_value.Resolve(),
	    .header = low_origin->header,
	    .initial_block = low_origin->initial_block,
	    .backedge_block = low_origin->backedge_block,
	    .read_block = read->Parent(),
	    .may_use_empty_mask_fallback = true,
	};
}

bool ProveWaterfallReadLaneUseGuard(const Program& program,
                                    const WaterfallReadLaneProof& proof,
                                    const Inst& use) {
	if (program.blocks.size() != program.block_info.size() || proof.header == nullptr ||
	    proof.initial_block == nullptr || proof.backedge_block == nullptr ||
	    proof.read_block == nullptr || use.Parent() == nullptr) {
		return false;
	}

	const auto header_index   = BlockIndex(program, proof.header);
	const auto backedge_index = BlockIndex(program, proof.backedge_block);
	const auto read_index     = BlockIndex(program, proof.read_block);
	const auto use_index      = BlockIndex(program, use.Parent());
	if (!header_index.has_value() || !backedge_index.has_value() ||
	    !read_index.has_value() || !use_index.has_value()) {
		return false;
	}

	const auto header_id = program.block_info[*header_index].id;
	const auto read_id   = program.block_info[*read_index].id;
	const auto use_id    = program.block_info[*use_index].id;

	const auto& header_term = program.block_info[*header_index].terminator;
	if (header_term.kind != CFG::TerminatorKind::Branch ||
	    header_term.true_block != read_id) {
		return false;
	}

	const auto& backedge_info = program.block_info[*backedge_index];
	if (backedge_info.terminator.kind != CFG::TerminatorKind::ConditionalBranch ||
	    backedge_info.terminator.true_block != header_id ||
	    !MatchAnyMaskNonZero(backedge_info.condition, proof.low_backedge,
	                         proof.high_backedge)) {
		return false;
	}

	const auto& read_info = program.block_info[*read_index];
	if (read_info.terminator.kind != CFG::TerminatorKind::ConditionalBranch) {
		return false;
	}

	bool use_guarded = false;
	if (read_info.terminator.true_block == use_id) {
		use_guarded = ConditionImpliesLoopCondition(
		    read_info.condition, proof.lane_condition, proof.header,
		    proof.backedge_block);
	} else if (read_info.terminator.false_block == use_id) {
		const auto condition = read_info.condition.Resolve();
		const auto* logical_not = condition.TryInstruction();
		if (logical_not != nullptr &&
		    logical_not->GetOpcode() == ValueOpcode::LogicalNot &&
		    logical_not->NumArgs() == 1u) {
			use_guarded = ConditionImpliesLoopCondition(
			    logical_not->Arg(0), proof.lane_condition, proof.header,
			    proof.backedge_block);
		}
	}
	if (!use_guarded) {
		return false;
	}

	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
