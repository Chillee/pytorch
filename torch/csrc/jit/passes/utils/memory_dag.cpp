#include "memory_dag.h"

#include <c10/util/flat_hash_map.h>
#include <torch/csrc/utils/memory.h>
#include <algorithm>
#include <queue>

namespace torch {
namespace jit {
namespace {
ska::flat_hash_map<const Element*, unsigned> comprMap;
ska::flat_hash_map<unsigned, const Element*> decomprMap;
} // namespace

unsigned Element::toIndex(const Element* x) {
  if (comprMap.count(x)) {
    return comprMap[x];
  }
  comprMap[x] = comprMap.size() + 1;
  decomprMap[comprMap.size()] = x;
  return comprMap[x];
}

const Element* Element::toElement(unsigned x) {
  auto res = decomprMap[x];
  TORCH_INTERNAL_ASSERT(res);
  return res;
}

bool MemoryDAG::mayAlias(Element* a, Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::mayAlias(const Element* a, const Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::mayAliasImpl(const Element* a, const Element* b) const {
  const auto aMemLoc = a->getMemoryLocations();
  const auto bMemLoc = b->getMemoryLocations();

  return aMemLoc.intersects(bMemLoc);
}

bool MemoryDAG::mayContainAlias(const Element* a, const Element* b) const {
  return mayContainAliasImpl(a, b);
}

bool MemoryDAG::mayContainAlias(Element* a, Element* b) const {
  return mayContainAliasImpl(a, b);
}

void collectAllContainedMemoryLocations(
    const Element* elem,
    MemoryLocations& cont) {
  // we have already recursed on this element
  unsigned compIdx = Element::toIndex(elem);
  if (cont.test(compIdx)) {
    return;
  }
  cont.set(compIdx);

  for (const auto& mem_loc : elem->getMemoryLocations()) {
    collectAllContainedMemoryLocations(Element::toElement(mem_loc), cont);
  }

  for (const auto& contained : elem->contained_elements) {
    collectAllContainedMemoryLocations(Element::toElement(contained), cont);
  }
}

bool MemoryDAG::mayContainAliasImpl(const Element* a, const Element* b) const {
  MemoryLocations all_a_mlocs;
  MemoryLocations all_b_mlocs;

  collectAllContainedMemoryLocations(a, all_a_mlocs);
  collectAllContainedMemoryLocations(b, all_b_mlocs);

  return all_a_mlocs.intersects(all_b_mlocs);
}

bool MemoryDAG::mayContainAlias(
    const at::ArrayRef<Element*>& a,
    const at::ArrayRef<Element*>& b) const {
  if (a.size() == 0 || b.size() == 0) {
    return false;
  }

  MemoryLocations all_a_mlocs;
  for (const auto& elem : a) {
    collectAllContainedMemoryLocations(elem, all_a_mlocs);
  }

  MemoryLocations all_b_mlocs;
  for (const auto& elem : b) {
    collectAllContainedMemoryLocations(elem, all_b_mlocs);
  }

  return all_a_mlocs.intersects(all_b_mlocs);
}

// Make `v` point at `to`.
void MemoryDAG::makePointerTo(Element* from, Element* to) {
  from->pointsTo.set(Element::toIndex(to));
  to->pointedFrom.set(Element::toIndex(from));
}

void MemoryDAG::addToContainedElements(Element* elem, Element* container) {
  container->contained_elements.set(Element::toIndex(elem));
}

// Give `v` a fresh alias (i.e. it does not point to any value)
Element* MemoryDAG::makeFreshValue(const Value* v) {
  auto el = torch::make_unique<Element>();
  el->value = v;

  auto rawPtr = el.get();
  elements_.emplace(rawPtr, std::move(el));
  return rawPtr;
}

const MemoryLocations& Element::getMemoryLocations() const {
  if (!cachedMemoryLocations_.empty()) {
    return cachedMemoryLocations_;
  }

  // Do a BFS in the `points-to` direction, collecting all memory locations
  MemoryLocations ret;
  this->bfs(BfsDirection::POINTS_TO, ret);
  cachedMemoryLocations_ = ret;
  return cachedMemoryLocations_;
}

// Do a breadth-first search over the graph, starting at `this` and
// traversing in the direction `dir`.`fn` will be run on each element.
void Element::bfs(BfsDirection dir, MemoryLocations& res) const {
  std::queue<unsigned> queue;
  MemoryLocations seen;
  queue.push(Element::toIndex(this));
  while (!queue.empty()) {
    const auto el = queue.front();
    queue.pop();
    seen.set(el);
    auto decompEl = Element::toElement(el);
    if (decompEl->pointsTo.empty()) {
      res.set(el);
    }

    switch (dir) {
      case BfsDirection::POINTS_TO: {
        for (auto ptr : decompEl->pointsTo) {
          if (!seen.test(ptr)) {
            queue.push(ptr);
          }
        }
      } break;

      case BfsDirection::POINTED_FROM: {
        for (auto ptr : decompEl->pointedFrom) {
          if (!seen.test(ptr)) {
            queue.push(ptr);
          }
        }
      } break;
    }
  }
}
} // namespace jit
} // namespace torch
