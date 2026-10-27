#pragma once

#include "py_table.hpp"
#include "mymyr/task/slot_map.hpp"

namespace mymyr::python
{
struct PySlotMap
{
    std::vector<TaskPtr> sources, targets;
    std::vector<std::shared_ptr<SlotMap>> maps;
    nb::object source, target;
    u32 source_words = 0, source_numeric = 0, target_words = 0, target_numeric = 0;
};

nb::object translate_task(nb::handle source, nb::handle states, nb::handle target, std::string_view missing,
                          nb::handle pool, nb::handle status, nb::handle stream, nb::handle ctx);
nb::object translate_table(nb::handle source, nb::handle rows, nb::handle ids, nb::handle target,
                           std::string_view missing, nb::handle pool, nb::handle status, nb::handle stream, nb::handle ctx);
void bind_translate(nb::module_& m);
#if defined(MYMYR_HAS_CUDA)
nb::object translate_device(PySlotMap& map, nb::handle rows, nb::handle ids, bool drop, nb::handle status,
                            nb::handle stream, nb::handle ctx);
#endif
}
