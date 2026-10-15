//! Acceleration structures: AABB and SAH BVH.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::MAX),
        max: Vec3::splat(f32::MIN),
    };

    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn from_points<'a>(points: impl IntoIterator<Item = &'a Vec3>) -> Self {
        let mut bb = Self::EMPTY;
        for p in points {
            bb.extend(*p);
        }
        bb
    }

    #[inline]
    pub fn extend(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    #[inline]
    pub fn union(&self, other: &AABB) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    #[inline]
    pub fn extents(&self) -> Vec3 {
        self.max - self.min
    }
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let e = self.extents();
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }

    #[inline]
    pub fn intersect(&self, origin: Vec3, inv_dir: Vec3) -> (f32, f32) {
        let t0 = (self.min - origin) * inv_dir;
        let t1 = (self.max - origin) * inv_dir;
        let tmin = t0.min(t1);
        let tmax = t0.max(t1);
        (
            tmin.x.max(tmin.y).max(tmin.z),
            tmax.x.min(tmax.y).min(tmax.z),
        )
    }
}

/// A BVH node, flattened for GPU transfer. Must match WGSL struct byte-for-byte.
/// WGSL layout (48 bytes):
///   bbox_min: vec3<f32> (12)  left_first: u32 (4)  → 16
///   bbox_max: vec3<f32> (12)  right_child: u32 (4) → 16
///   prim_count: u32 (4)       _pad: vec3<f32> (12) → 16
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BVHNode {
    pub bbox_min: [f32; 3],
    pub left_first: u32, // left child or first primitive index
    pub bbox_max: [f32; 3],
    pub right_child: u32, // right child index (internal) or unused (leaf)
    pub prim_count: u32,  // 0 for internal, leaf primitive count
    pub _pad: [f32; 3],   // pad to 48 bytes
}

impl BVHNode {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.prim_count > 0
    }
}

#[derive(Debug, Clone)]
pub struct BVH {
    pub nodes: Vec<BVHNode>,
    pub prim_indices: Vec<u32>,
}

/// One SAH bin: the primitives routed into it, summarized as a count plus the
/// union of their AABBs (so split costs use real surface area, not centroids).
#[derive(Clone, Copy)]
struct Bin {
    bounds: AABB,
    count: usize,
}

impl Bin {
    fn empty() -> Self {
        Self {
            bounds: AABB::EMPTY,
            count: 0,
        }
    }
}

pub struct BVHBuilder {
    centroids: Vec<Vec3>,
    bboxes: Vec<AABB>,
    max_prims_per_leaf: usize,
}

impl BVHBuilder {
    pub fn new(centroids: Vec<Vec3>, bboxes: Vec<AABB>) -> Self {
        Self {
            centroids,
            bboxes,
            max_prims_per_leaf: 4,
        }
    }

    pub fn build(&self) -> BVH {
        let mut nodes: Vec<BVHNode> = Vec::new();
        let mut prim_indices: Vec<u32> = (0..self.centroids.len() as u32).collect();
        let len = prim_indices.len();

        // Build children-first so siblings are consecutive, then parent after.
        // Root ends up at the last index.
        self.build_subtree(&mut nodes, &mut prim_indices, 0, len);

        // Reverse so root is at index 0, and remap child indices.
        Self::reverse_nodes(&mut nodes);

        BVH {
            nodes,
            prim_indices,
        }
    }

    fn build_subtree(
        &self,
        nodes: &mut Vec<BVHNode>,
        prims: &mut [u32],
        start: usize,
        end: usize,
    ) -> u32 {
        let count = end - start;
        if count == 0 {
            return 0;
        }

        let mut bb = AABB::EMPTY;
        for i in start..end {
            bb = bb.union(&self.bboxes[prims[i] as usize]);
        }

        // Only bother splitting when a leaf would exceed the size limit.
        let split = if count > self.max_prims_per_leaf {
            self.find_best_split(prims, start, end, &bb)
        } else {
            None
        };

        let Some((axis, left_count)) = split else {
            // A leaf is cheaper (or geometry is degenerate) — emit one node.
            let idx = nodes.len() as u32;
            nodes.push(BVHNode {
                bbox_min: bb.min.to_array(),
                left_first: start as u32,
                bbox_max: bb.max.to_array(),
                right_child: 0,
                prim_count: count as u32,
                _pad: [0.0; 3],
            });
            return idx;
        };

        // Partition prims[start..end] by centroid[axis]: the first `left_count`
        // elements (smallest centroids) become the left child's range.
        let mid = start + left_count;
        prims[start..end].select_nth_unstable_by(left_count, |&a, &b| {
            let ca = self.centroids[a as usize][axis];
            let cb = self.centroids[b as usize][axis];
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build children first — they will occupy consecutive indices
        let left_idx = self.build_subtree(nodes, prims, start, mid);
        let right_idx = self.build_subtree(nodes, prims, mid, end);
        // right_idx should be left_idx + 1 (siblings are consecutive)

        let idx = nodes.len() as u32;
        nodes.push(BVHNode {
            bbox_min: bb.min.to_array(),
            left_first: left_idx,
            bbox_max: bb.max.to_array(),
            right_child: right_idx,
            prim_count: 0, // internal node marker
            _pad: [0.0; 3],
        });
        idx
    }

    /// Binned SAH: pick the axis+split with the lowest surface-area cost.
    ///
    /// Primitives are assigned to `SAH_BINS` buckets by centroid along each
    /// axis; each bucket accumulates the *full* primitive AABBs so costs use
    /// real surface area rather than centroid positions. Returns
    /// `Some((axis, left_count))` when a split is cheaper than a leaf, else
    /// `None` (meaning "make a leaf").
    fn find_best_split(
        &self,
        prims: &[u32],
        start: usize,
        end: usize,
        bb: &AABB,
    ) -> Option<(usize, usize)> {
        const SAH_BINS: usize = 12;

        let bb_sa = bb.surface_area();
        if bb_sa <= 0.0 {
            return None;
        } // fully degenerate box: nothing to split
        let inv_bb_sa = 1.0 / bb_sa;

        // Binning operates on *centroid* positions, so compute their range
        // (not the bbox range — centroids may legitimately sit near a box edge).
        let mut cmin = Vec3::splat(f32::MAX);
        let mut cmax = Vec3::splat(f32::MIN);
        for &p in &prims[start..end] {
            let c = self.centroids[p as usize];
            cmin = cmin.min(c);
            cmax = cmax.max(c);
        }

        let mut best_axis = 0usize;
        let mut best_left = 0usize;
        let mut best_cost = f32::INFINITY;

        for axis in 0..3 {
            let range = cmax[axis] - cmin[axis];
            if range <= 0.0 {
                continue;
            } // all centroids coincide on this axis

            // `scale` maps centroids onto [0, SAH_BINS); the last bin is clamped.
            let scale = SAH_BINS as f32 / range;
            let mut bins = [Bin::empty(); SAH_BINS];
            for &p in &prims[start..end] {
                let prim = p as usize;
                let c = self.centroids[prim][axis];
                let mut bin = ((c - cmin[axis]) * scale) as usize;
                if bin >= SAH_BINS {
                    bin = SAH_BINS - 1;
                } // c == cmax[axis]
                bins[bin].count += 1;
                bins[bin].bounds = bins[bin].bounds.union(&self.bboxes[prim]);
            }

            // Prefix (left→right) and suffix (right→left) accumulations, so the
            // cost of every candidate split is O(1) once the bins are built.
            let (mut left_area, mut left_count) = ([0.0f32; SAH_BINS], [0usize; SAH_BINS]);
            let (mut right_area, mut right_count) = ([0.0f32; SAH_BINS], [0usize; SAH_BINS]);
            let mut acc = AABB::EMPTY;
            let mut n = 0usize;
            for b in 0..SAH_BINS {
                acc = acc.union(&bins[b].bounds);
                n += bins[b].count;
                left_area[b] = acc.surface_area();
                left_count[b] = n;
            }
            let mut acc = AABB::EMPTY;
            let mut n = 0usize;
            for b in (0..SAH_BINS).rev() {
                acc = acc.union(&bins[b].bounds);
                n += bins[b].count;
                right_area[b] = acc.surface_area();
                right_count[b] = n;
            }

            // A split sits *between* bin b and b+1: left = bins 0..=b, right = b+1..
            for b in 0..SAH_BINS - 1 {
                let lc = left_count[b];
                let rc = right_count[b + 1];
                if lc == 0 || rc == 0 {
                    continue;
                } // empty side is never optimal
                let cost =
                    1.0 + (left_area[b] * lc as f32 + right_area[b + 1] * rc as f32) * inv_bb_sa;
                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_left = lc;
                }
            }
        }

        // Leaf cost ≈ one intersection test per primitive; split cost ≈ one
        // traversal step plus the two children's surface-area terms.
        let leaf_cost = (end - start) as f32;
        if best_cost < leaf_cost {
            Some((best_axis, best_left))
        } else {
            None
        }
    }

    /// Reverse node order so root is index 0, remapping child indices.
    fn reverse_nodes(nodes: &mut Vec<BVHNode>) {
        let n = nodes.len();
        if n <= 1 {
            return;
        }
        nodes.reverse();
        // Remap: old_idx → new_idx = n - 1 - old_idx
        for node in nodes.iter_mut() {
            if !node.is_leaf() {
                node.left_first = n as u32 - 1 - node.left_first;
                node.right_child = n as u32 - 1 - node.right_child;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_empty() {
        assert!(AABB::EMPTY.min.x > AABB::EMPTY.max.x);
    }

    #[test]
    fn test_aabb_intersect() {
        let bb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let (t0, t1) = bb.intersect(Vec3::new(0.0, 0.0, -3.0), Vec3::new(0.0, 0.0, 1.0).recip());
        assert!(t0 <= t1 && t0 > 0.0);
    }

    #[test]
    fn test_bvh_build_empty() {
        let bvh = BVHBuilder::new(vec![], vec![]).build();
        assert!(bvh.nodes.is_empty());
    }

    #[test]
    fn test_bvh_build_single() {
        let centroids = vec![Vec3::ZERO];
        let bboxes = vec![AABB::new(Vec3::splat(-0.5), Vec3::splat(0.5))];
        let bvh = BVHBuilder::new(centroids, bboxes).build();
        assert_eq!(bvh.nodes.len(), 1);
        assert!(bvh.nodes[0].is_leaf());
    }

    #[test]
    fn test_bvh_build_many() {
        let mut centroids = vec![];
        let mut bboxes = vec![];
        for i in 0..12 {
            let x = (i % 4) as f32;
            let y = (i / 4) as f32;
            centroids.push(Vec3::new(x, y, 0.0));
            // Non-overlapping boxes centered on their centroid — the realistic
            // invariant (a triangle's centroid lies inside its own AABB).
            bboxes.push(AABB::new(
                Vec3::new(x - 0.1, y - 0.1, -0.1),
                Vec3::new(x + 0.1, y + 0.1, 0.1),
            ));
        }
        let bvh = BVHBuilder::new(centroids, bboxes).build();
        assert!(bvh.nodes.len() > 1);
        // Root is index 0, should be internal
        assert!(!bvh.nodes[0].is_leaf());
        // Verify child indices are valid
        assert!(bvh.nodes[0].left_first < bvh.nodes.len() as u32);
        assert!(bvh.nodes[0].right_child < bvh.nodes.len() as u32);
    }

    /// Iterative traversal returning every primitive index a leaf references.
    fn collect_prims(bvh: &BVH) -> Vec<u32> {
        let mut out = Vec::new();
        let mut stack = vec![0u32];
        while let Some(idx) = stack.pop() {
            assert!(
                idx < bvh.nodes.len() as u32,
                "child index out of range: {idx}"
            );
            let node = &bvh.nodes[idx as usize];
            if node.is_leaf() {
                for i in 0..node.prim_count {
                    out.push(bvh.prim_indices[(node.left_first + i) as usize]);
                }
            } else {
                stack.push(node.right_child);
                stack.push(node.left_first);
            }
        }
        out
    }

    #[test]
    fn test_bvh_sah_enumerates_all_primitives_once() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let n = 513; // odd count exercises uneven splits
        let mut rng = StdRng::seed_from_u64(1234);
        let mut centroids = Vec::with_capacity(n);
        let mut bboxes = Vec::with_capacity(n);
        for _ in 0..n {
            let c = Vec3::new(
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            );
            centroids.push(c);
            bboxes.push(AABB::new(c - Vec3::splat(0.05), c + Vec3::splat(0.05)));
        }
        let bvh = BVHBuilder::new(centroids, bboxes).build();

        let mut seen = collect_prims(&bvh);
        seen.sort_unstable();
        let mut expected: Vec<u32> = (0..n as u32).collect();
        expected.sort_unstable();
        assert_eq!(
            seen, expected,
            "every primitive must be reachable exactly once"
        );
    }

    #[test]
    fn test_bvh_sah_coincident_centroids_still_terminates() {
        // Identical centroids => no split can separate them; the builder must
        // give up and emit one (large) leaf instead of recursing forever.
        let n = 16;
        let centroids = vec![Vec3::ZERO; n];
        let bboxes = vec![AABB::new(Vec3::splat(-0.5), Vec3::splat(0.5)); n];
        let bvh = BVHBuilder::new(centroids, bboxes).build();
        assert_eq!(
            bvh.nodes.len(),
            1,
            "degenerate input should collapse to one leaf"
        );
        assert!(bvh.nodes[0].is_leaf());
        assert_eq!(bvh.nodes[0].prim_count, n as u32);
    }
}
