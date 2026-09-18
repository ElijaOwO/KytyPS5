#include "graphics/shader/recompiler/ir/passes/ResourceKeyAnalysis.h"

#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint32_t MaxAnalysisDepth = 96u;

struct ConditionAssumption {
	Value condition;
	bool  value = true;
};

struct AnalysisContext {
	std::vector<ConditionAssumption> assumptions;
	bool selected_lane_satisfies_condition = false;
};

struct F32Range {
	float minimum = 0.0f;
	float maximum = 0.0f;
	bool  has_numeric = false;
	bool  may_nan = false;
};

enum class TruthValue { False, True, Unknown };

bool ConditionImplies(Value premise, Value target, uint32_t depth = 0) {
	if (depth >= MaxAnalysisDepth) {
		return false;
	}
	premise = premise.Resolve();
	target  = target.Resolve();
	if (premise == target) {
		return true;
	}
	const auto* inst = premise.TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LogicalAnd ||
	    inst->NumArgs() != 2u) {
		return false;
	}
	return ConditionImplies(inst->Arg(0), target, depth + 1u) ||
	       ConditionImplies(inst->Arg(1), target, depth + 1u);
}

TruthValue AssumedTruth(Value condition, const AnalysisContext& context) {
	condition = condition.Resolve();
	if (condition.IsImmediate() && condition.GetType() == Type::U1) {
		return condition.U1() ? TruthValue::True : TruthValue::False;
	}
	for (const auto& assumption: context.assumptions) {
		if (assumption.condition.Resolve() == condition) {
			return assumption.value ? TruthValue::True : TruthValue::False;
		}
		if (assumption.value && ConditionImplies(assumption.condition, condition)) {
			return TruthValue::True;
		}
	}
	const auto* inst = condition.TryInstruction();
	if (inst != nullptr && inst->GetOpcode() == ValueOpcode::LogicalNot &&
	    inst->NumArgs() == 1u) {
		const auto nested = AssumedTruth(inst->Arg(0), context);
		if (nested == TruthValue::True) {
			return TruthValue::False;
		}
		if (nested == TruthValue::False) {
			return TruthValue::True;
		}
	}
	return TruthValue::Unknown;
}

std::optional<F32Range> AnalyzeF32(Value value, const AnalysisContext& context,
                                   uint32_t depth);
std::optional<U32Range> AnalyzeU32(Value value, const AnalysisContext& context,
                                   uint32_t depth);
TruthValue AnalyzeCondition(Value value, const AnalysisContext& context, uint32_t depth);

AnalysisContext WithAssumption(const AnalysisContext& context, Value condition, bool value) {
	auto next = context;
	next.assumptions.push_back({condition.Resolve(), value});
	return next;
}

F32Range Join(const F32Range& left, const F32Range& right) {
	F32Range result;
	result.has_numeric = left.has_numeric || right.has_numeric;
	result.may_nan     = left.may_nan || right.may_nan;
	if (left.has_numeric && right.has_numeric) {
		result.minimum = std::min(left.minimum, right.minimum);
		result.maximum = std::max(left.maximum, right.maximum);
	} else if (left.has_numeric) {
		result.minimum = left.minimum;
		result.maximum = left.maximum;
	} else if (right.has_numeric) {
		result.minimum = right.minimum;
		result.maximum = right.maximum;
	}
	return result;
}

std::optional<U32Range> Join(const U32Range& left, const U32Range& right) {
	return U32Range {
	    .minimum = std::min(left.minimum, right.minimum),
	    .maximum = std::max(left.maximum, right.maximum),
	};
}

void ApplyNanAssumptions(Value value, const AnalysisContext& context, F32Range& range) {
	value = value.Resolve();
	for (const auto& assumption: context.assumptions) {
		if (assumption.value) {
			continue;
		}
		const auto* inst = assumption.condition.Resolve().TryInstruction();
		if (inst != nullptr && inst->GetOpcode() == ValueOpcode::FPIsNan32 &&
		    inst->NumArgs() == 1u && inst->Arg(0).Resolve() == value) {
			range.may_nan = false;
		}
	}
}

Value ResolveConditionalU32Bits(Value value, const AnalysisContext& context, uint32_t depth) {
	for (; depth < MaxAnalysisDepth; depth++) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::SelectU32 ||
		    inst->NumArgs() != 3u) {
			return value;
		}
		const auto truth = AnalyzeCondition(inst->Arg(0), context, depth + 1u);
		if (truth == TruthValue::True) {
			value = inst->Arg(1);
		} else if (truth == TruthValue::False) {
			value = inst->Arg(2);
		} else {
			return value;
		}
	}
	return value.Resolve();
}

std::optional<F32Range> AnalyzeF32Binary(Value left_value, Value right_value,
                                         const AnalysisContext& context, uint32_t depth,
                                         ValueOpcode opcode) {
	const auto left  = AnalyzeF32(left_value, context, depth + 1u);
	const auto right = AnalyzeF32(right_value, context, depth + 1u);
	if (!left.has_value() || !right.has_value()) {
		return std::nullopt;
	}
	F32Range result;
	result.may_nan = left->may_nan || right->may_nan;
	if (!left->has_numeric || !right->has_numeric) {
		return result;
	}
	std::array<float, 4> values {};
	switch (opcode) {
		case ValueOpcode::FPAdd32:
			values = {left->minimum + right->minimum, left->minimum + right->maximum,
			          left->maximum + right->minimum, left->maximum + right->maximum};
			break;
		case ValueOpcode::FPSub32:
			values = {left->minimum - right->minimum, left->minimum - right->maximum,
			          left->maximum - right->minimum, left->maximum - right->maximum};
			break;
		case ValueOpcode::FPMul32:
			values = {left->minimum * right->minimum, left->minimum * right->maximum,
			          left->maximum * right->minimum, left->maximum * right->maximum};
			break;
		default: return std::nullopt;
	}
	if (std::ranges::any_of(values, [](float v) { return std::isnan(v); })) {
		result.may_nan = true;
	}
	auto finite = values | std::views::filter([](float v) { return !std::isnan(v); });
	auto begin  = finite.begin();
	if (begin == finite.end()) {
		return result;
	}
	result.has_numeric = true;
	result.minimum     = *begin;
	result.maximum     = *begin;
	for (const auto value: finite) {
		result.minimum = std::min(result.minimum, value);
		result.maximum = std::max(result.maximum, value);
	}
	return result;
}

std::optional<F32Range> AnalyzeF32(Value value, const AnalysisContext& context,
                                   uint32_t depth) {
	if (depth >= MaxAnalysisDepth) {
		return std::nullopt;
	}
	value = value.Resolve();
	if (value.IsImmediate()) {
		if (value.GetType() != Type::F32) {
			return std::nullopt;
		}
		const auto immediate = value.F32Value();
		if (std::isnan(immediate)) {
			return F32Range {.has_numeric = false, .may_nan = true};
		}
		return F32Range {.minimum = immediate, .maximum = immediate, .has_numeric = true};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return std::nullopt;
	}

	std::optional<F32Range> result;
	switch (inst->GetOpcode()) {
		case ValueOpcode::BitCastF32U32: {
			const auto bits      = ResolveConditionalU32Bits(inst->Arg(0), context, depth + 1u);
			const auto* bit_inst = bits.TryInstruction();
			if (bit_inst == nullptr || bit_inst->GetOpcode() != ValueOpcode::BitCastU32F32 ||
			    bit_inst->NumArgs() != 1u) {
				return std::nullopt;
			}
			result = AnalyzeF32(bit_inst->Arg(0), context, depth + 1u);
			break;
		}
		case ValueOpcode::FPSaturate32:
			result = F32Range {
			    .minimum = 0.0f,
			    .maximum = 1.0f,
			    .has_numeric = true,
			    .may_nan = true,
			};
			break;
		case ValueOpcode::FPNeg32: {
			const auto source = AnalyzeF32(inst->Arg(0), context, depth + 1u);
			if (!source.has_value()) {
				return std::nullopt;
			}
			result = *source;
			if (result->has_numeric) {
				const auto minimum = -source->maximum;
				result->maximum   = -source->minimum;
				result->minimum   = minimum;
			}
			break;
		}
		case ValueOpcode::FPAbs32: {
			const auto source = AnalyzeF32(inst->Arg(0), context, depth + 1u);
			if (!source.has_value()) {
				return std::nullopt;
			}
			result = *source;
			if (result->has_numeric) {
				const auto maximum =
				    std::max(std::abs(source->minimum), std::abs(source->maximum));
				result->minimum = source->minimum <= 0.0f && source->maximum >= 0.0f
				                      ? 0.0f
				                      : std::min(std::abs(source->minimum), std::abs(source->maximum));
				result->maximum = maximum;
			}
			break;
		}
		case ValueOpcode::FPAdd32:
		case ValueOpcode::FPSub32:
		case ValueOpcode::FPMul32:
			result = AnalyzeF32Binary(inst->Arg(0), inst->Arg(1), context, depth,
			                          inst->GetOpcode());
			break;
		case ValueOpcode::FPFma32: {
			const auto left  = AnalyzeF32(inst->Arg(0), context, depth + 1u);
			const auto right = AnalyzeF32(inst->Arg(1), context, depth + 1u);
			const auto add   = AnalyzeF32(inst->Arg(2), context, depth + 1u);
			if (!left.has_value() || !right.has_value() || !add.has_value()) {
				return std::nullopt;
			}
			F32Range range;
			range.may_nan = left->may_nan || right->may_nan || add->may_nan;
			if (left->has_numeric && right->has_numeric && add->has_numeric) {
				std::array<float, 8> values {};
				size_t index = 0;
				for (const auto lhs: {left->minimum, left->maximum}) {
					for (const auto rhs: {right->minimum, right->maximum}) {
						for (const auto extra: {add->minimum, add->maximum}) {
							values[index++] = std::fma(lhs, rhs, extra);
						}
					}
				}
				range.has_numeric = true;
				range.minimum = values[0];
				range.maximum = values[0];
				for (const auto candidate: values) {
					if (std::isnan(candidate)) {
						range.may_nan = true;
						continue;
					}
					range.minimum = std::min(range.minimum, candidate);
					range.maximum = std::max(range.maximum, candidate);
				}
			}
			result = range;
			break;
		}
		case ValueOpcode::FPFloor32:
		case ValueOpcode::FPTrunc32: {
			const auto source = AnalyzeF32(inst->Arg(0), context, depth + 1u);
			if (!source.has_value()) {
				return std::nullopt;
			}
			result = *source;
			if (result->has_numeric) {
				if (inst->GetOpcode() == ValueOpcode::FPFloor32) {
					result->minimum = std::floor(source->minimum);
					result->maximum = std::floor(source->maximum);
				} else {
					result->minimum = std::trunc(source->minimum);
					result->maximum = std::trunc(source->maximum);
				}
			}
			break;
		}
		case ValueOpcode::SelectF32: {
			const auto truth = AnalyzeCondition(inst->Arg(0), context, depth + 1u);
			if (truth == TruthValue::True) {
				result = AnalyzeF32(inst->Arg(1), context, depth + 1u);
			} else if (truth == TruthValue::False) {
				result = AnalyzeF32(inst->Arg(2), context, depth + 1u);
			} else {
				const auto true_range =
				    AnalyzeF32(inst->Arg(1), WithAssumption(context, inst->Arg(0), true),
				               depth + 1u);
				const auto false_range =
				    AnalyzeF32(inst->Arg(2), WithAssumption(context, inst->Arg(0), false),
				               depth + 1u);
				if (!true_range.has_value() || !false_range.has_value()) {
					return std::nullopt;
				}
				result = Join(*true_range, *false_range);
			}
			break;
		}
		default: return std::nullopt;
	}
	if (result.has_value()) {
		ApplyNanAssumptions(value, context, *result);
	}
	return result;
}

TruthValue AnalyzeOrderedComparison(Value left_value, Value right_value,
                                    const AnalysisContext& context, uint32_t depth,
                                    ValueOpcode opcode) {
	const auto left  = AnalyzeF32(left_value, context, depth + 1u);
	const auto right = AnalyzeF32(right_value, context, depth + 1u);
	if (!left.has_value() || !right.has_value()) {
		return TruthValue::Unknown;
	}
	if (!left->has_numeric || !right->has_numeric) {
		return TruthValue::False;
	}
	const bool may_nan = left->may_nan || right->may_nan;
	switch (opcode) {
		case ValueOpcode::FPOrdLessThan32:
			if (left->minimum >= right->maximum) return TruthValue::False;
			if (!may_nan && left->maximum < right->minimum) return TruthValue::True;
			break;
		case ValueOpcode::FPOrdLessThanEqual32:
			if (left->minimum > right->maximum) return TruthValue::False;
			if (!may_nan && left->maximum <= right->minimum) return TruthValue::True;
			break;
		case ValueOpcode::FPOrdGreaterThan32:
			if (left->maximum <= right->minimum) return TruthValue::False;
			if (!may_nan && left->minimum > right->maximum) return TruthValue::True;
			break;
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (left->maximum < right->minimum) return TruthValue::False;
			if (!may_nan && left->minimum >= right->maximum) return TruthValue::True;
			break;
		default: break;
	}
	return TruthValue::Unknown;
}

TruthValue AnalyzeCondition(Value value, const AnalysisContext& context, uint32_t depth) {
	if (depth >= MaxAnalysisDepth) {
		return TruthValue::Unknown;
	}
	const auto assumed = AssumedTruth(value, context);
	if (assumed != TruthValue::Unknown) {
		return assumed;
	}
	value            = value.Resolve();
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return TruthValue::Unknown;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::FPIsNan32: {
			const auto range = AnalyzeF32(inst->Arg(0), context, depth + 1u);
			if (!range.has_value()) {
				return TruthValue::Unknown;
			}
			if (!range->may_nan) {
				return TruthValue::False;
			}
			if (!range->has_numeric) {
				return TruthValue::True;
			}
			return TruthValue::Unknown;
		}
		case ValueOpcode::FPOrdLessThan32:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThan32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			return AnalyzeOrderedComparison(inst->Arg(0), inst->Arg(1), context, depth,
			                                inst->GetOpcode());
		case ValueOpcode::LogicalNot: {
			const auto nested = AnalyzeCondition(inst->Arg(0), context, depth + 1u);
			if (nested == TruthValue::True) return TruthValue::False;
			if (nested == TruthValue::False) return TruthValue::True;
			return TruthValue::Unknown;
		}
		default: return TruthValue::Unknown;
	}
}

std::optional<U32Range> AnalyzeU32(Value value, const AnalysisContext& context,
                                   uint32_t depth) {
	if (depth >= MaxAnalysisDepth) {
		return std::nullopt;
	}
	value = value.Resolve();
	if (value.IsImmediate()) {
		if (value.GetType() != Type::U32) {
			return std::nullopt;
		}
		return U32Range {.minimum = value.U32(), .maximum = value.U32()};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return std::nullopt;
	}

	switch (inst->GetOpcode()) {
		case ValueOpcode::SelectU32: {
			const auto truth = AnalyzeCondition(inst->Arg(0), context, depth + 1u);
			if (truth == TruthValue::True) {
				return AnalyzeU32(inst->Arg(1), context, depth + 1u);
			}
			if (truth == TruthValue::False) {
				return AnalyzeU32(inst->Arg(2), context, depth + 1u);
			}
			const auto true_range =
			    AnalyzeU32(inst->Arg(1), WithAssumption(context, inst->Arg(0), true), depth + 1u);
			const auto false_range =
			    AnalyzeU32(inst->Arg(2), WithAssumption(context, inst->Arg(0), false),
			               depth + 1u);
			if (!true_range.has_value() || !false_range.has_value()) {
				return std::nullopt;
			}
			return Join(*true_range, *false_range);
		}
		case ValueOpcode::ReadLane: {
			auto source_context = context;
			if (!context.selected_lane_satisfies_condition) {
				source_context.assumptions.clear();
			}
			source_context.selected_lane_satisfies_condition = false;
			return AnalyzeU32(inst->Arg(0), source_context, depth + 1u);
		}
		case ValueOpcode::IAdd32: {
			const auto left  = AnalyzeU32(inst->Arg(0), context, depth + 1u);
			const auto right = AnalyzeU32(inst->Arg(1), context, depth + 1u);
			if (!left.has_value() || !right.has_value()) {
				return std::nullopt;
			}
			const auto minimum = static_cast<uint64_t>(left->minimum) + right->minimum;
			const auto maximum = static_cast<uint64_t>(left->maximum) + right->maximum;
			if (maximum > UINT32_MAX) {
				return std::nullopt;
			}
			return U32Range {
			    .minimum = static_cast<uint32_t>(minimum),
			    .maximum = static_cast<uint32_t>(maximum),
			};
		}
		case ValueOpcode::UMin32: {
			const auto left  = AnalyzeU32(inst->Arg(0), context, depth + 1u);
			const auto right = AnalyzeU32(inst->Arg(1), context, depth + 1u);
			if (!left.has_value() || !right.has_value()) {
				return std::nullopt;
			}
			return U32Range {
			    .minimum = std::min(left->minimum, right->minimum),
			    .maximum = std::min(left->maximum, right->maximum),
			};
		}
		case ValueOpcode::SMin32: {
			const auto left  = AnalyzeU32(inst->Arg(0), context, depth + 1u);
			const auto right = AnalyzeU32(inst->Arg(1), context, depth + 1u);
			if (!left.has_value() || !right.has_value() ||
			    left->maximum > static_cast<uint32_t>(INT32_MAX) ||
			    right->maximum > static_cast<uint32_t>(INT32_MAX)) {
				return std::nullopt;
			}
			return U32Range {
			    .minimum = std::min(left->minimum, right->minimum),
			    .maximum = std::min(left->maximum, right->maximum),
			};
		}
		case ValueOpcode::ShiftLeftLogical32: {
			const auto source = AnalyzeU32(inst->Arg(0), context, depth + 1u);
			const auto shift  = inst->Arg(1).Resolve();
			if (!source.has_value() || !shift.IsImmediate() || shift.GetType() != Type::U32) {
				return std::nullopt;
			}
			const auto amount = shift.U32() & 31u;
			if (source->maximum > (UINT32_MAX >> amount)) {
				return std::nullopt;
			}
			return U32Range {
			    .minimum = source->minimum << amount,
			    .maximum = source->maximum << amount,
			};
		}
		case ValueOpcode::ConvertS32F32: {
			const auto source = AnalyzeF32(inst->Arg(0), context, depth + 1u);
			if (!source.has_value() || !source->has_numeric || source->may_nan ||
			    source->minimum < 0.0f ||
			    source->maximum > static_cast<float>(INT32_MAX)) {
				return std::nullopt;
			}
			return U32Range {
			    .minimum = static_cast<uint32_t>(static_cast<int32_t>(source->minimum)),
			    .maximum = static_cast<uint32_t>(static_cast<int32_t>(source->maximum)),
			};
		}
		case ValueOpcode::BitCastU32F32: {
			const auto source = inst->Arg(0).Resolve();
			const auto* source_inst = source.TryInstruction();
			if (source_inst != nullptr &&
			    source_inst->GetOpcode() == ValueOpcode::BitCastF32U32 &&
			    source_inst->NumArgs() == 1u) {
				return AnalyzeU32(source_inst->Arg(0), context, depth + 1u);
			}
			return std::nullopt;
		}
		default: return std::nullopt;
	}
}

} // namespace

std::optional<U32Range> AnalyzeResourceKeyRange(Value value, ResourceKeyRangeContext context) {
	AnalysisContext analysis;
	if (!context.condition.IsEmpty()) {
		analysis.assumptions.push_back({context.condition.Resolve(), true});
	}
	analysis.selected_lane_satisfies_condition = context.selected_lane_satisfies_condition;
	return AnalyzeU32(value, analysis, 0u);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
