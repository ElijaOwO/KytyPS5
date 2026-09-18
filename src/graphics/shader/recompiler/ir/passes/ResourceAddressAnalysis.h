#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEADDRESSANALYSIS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEADDRESSANALYSIS_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceKeyAnalysis.h"

#include <array>
#include <cstdint>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct AddressIndirectImageAnalysis {
	Value                   key;
	Value                   byte_offset;
	Value                   lane_condition;
	Inst*                    address_handle = nullptr;
	std::array<const Inst*, 8> reads {};
	std::array<uint32_t, 8>    memory {};
	U32Range                conditional_key_range;
	uint32_t                descriptor_stride = 0;
	uint32_t                candidate_count = 0;

	// The key range is valid when the recognized waterfall selects a set bit.
	// Empty-mask fallback remains explicit until control-flow proves it unreachable.
	bool requires_nonempty_wave_mask = true;
};

std::optional<AddressIndirectImageAnalysis> AnalyzeAddressIndirectImage(
    const Program& program, const Inst& handle);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEADDRESSANALYSIS_H_ */
