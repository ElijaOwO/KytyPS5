#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEKEYANALYSIS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEKEYANALYSIS_H_

#include "graphics/shader/recompiler/ir/Value.h"

#include <cstdint>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct U32Range {
	uint32_t minimum = 0;
	uint32_t maximum = 0;

	bool operator==(const U32Range& other) const = default;
};

struct ResourceKeyRangeContext {
	Value condition;
	bool  selected_lane_satisfies_condition = false;
};

// Computes a conservative non-wrapping U32 interval for a resource key.
//
// condition is an assumption known to hold for the lane whose value is being analyzed.
// ReadLane only inherits that assumption when selected_lane_satisfies_condition is true;
// callers must establish that property from control-flow/wave provenance first.
std::optional<U32Range> AnalyzeResourceKeyRange(
    Value value, ResourceKeyRangeContext context = {});

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEKEYANALYSIS_H_ */
