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
    pub const EMPTY: Self = Self { min: Vec3::splat(f32::MAX), max: Vec3::splat(f32::MIN) };

    #[inline] pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    #[inline]
    pub fn from_points<'a>(points: impl IntoIterator<Item = &'a Vec3>) -> Self {
        let mut bb = Self::EMPTY;
        for p in points { bb.extend(*p); }
        bb
    }

    #[inline] pub fn extend(&mut self, p: Vec3) { self.min = self.min.min(p); self.max = self.max.max(p); }

    #[inline]
    pub fn union(&self, other: &AABB) -> Self {
        Self { min: self.min.min(other.min), max: self.max.max(other.max) }
    }

    #[inline] pub fn center(&self) -> Vec3 { (self.min + self.max) * 0.5 }
    #[inline] pub fn extents(&self) -> Vec3 { self.max - self.min }
    #[inline] pub fn surface_area(&self) -> f32 {
        let e = self.extents(); 2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }

    #[inline]
    pub fn intersect(&self, origin: Vec3, inv_dir: Vec3) -> (f32, f32) {
        let t0 = (self.min - origin) * inv_dir;
        let t1 = (self.max - origin) * inv_dir;
        let tmin = t0.min(t1);
        let tmax = t0.max(t1);
        (tmin.x.max(tmin.y).max(tmin.z), tmax.x.min(tmax.y).min(tmax.z))
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
    pub left_first: u32,     // left child or first primitive index
    pub bbox_max: [f32; 3],
    pub right_child: u32,    // right child index (internal) or unused (leaf)
    pub prim_count: u32,     // 0 for internal, leaf primitive count
    pub _pad: [f32; 3],      // pad to 48 bytes
}

impl BVHNode {
    #[inline] pub fn is_leaf(&self) -> bool { self.prim_count > 0 }
}

#[derive(Debug, Clone)]
pub struct BVH {
    pub nodes: Vec<BVHNode>,
    pub prim_indices: Vec<u32>,
}

pub struct BVHBuilder {
    centroids: Vec<Vec3>,
    bboxes: Vec<AABB>,
    max_prims_per_leaf: usize,
}

impl BVHBuilder {
    pub fn new(centroids: Vec<Vec3>, bboxes: Vec<AABB>) -> Self {
        Self { centroids, bboxes, max_prims_per_leaf: 4 }
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

        BVH { nodes, prim_indices }
    }

    fn build_subtree(
        &self,
        nodes: &mut Vec<BVHNode>,
        prims: &mut [u32],
        start: usize,
        end: usize,
    ) -> u32 {
        let count = end - start;
        if count == 0 { return 0; }

        let mut bb = AABB::EMPTY;
        for i in start..end { bb = bb.union(&self.bboxes[prims[i] as usize]); }

        if count <= self.max_prims_per_leaf {
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
        }

        let extents = bb.extents();
        let axis = if extents.x >= extents.y && extents.x >= extents.z { 0 }
                   else if extents.y >= extents.z { 1 } else { 2 };

        let mid = (start + end) / 2;
        prims[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
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

    /// Reverse node order so root is index 0, remapping child indices.
    fn reverse_nodes(nodes: &mut Vec<BVHNode>) {
        let n = nodes.len();
        if n <= 1 { return; }
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
    fn test_aabb_empty() { assert!(AABB::EMPTY.min.x > AABB::EMPTY.max.x); }

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
            bboxes.push(AABB::new(Vec3::splat(-0.1), Vec3::splat(0.1)));
        }
        let bvh = BVHBuilder::new(centroids, bboxes).build();
        assert!(bvh.nodes.len() > 1);
        // Root is index 0, should be internal
        assert!(!bvh.nodes[0].is_leaf());
        // Verify child indices are valid
        assert!(bvh.nodes[0].left_first < bvh.nodes.len() as u32);
        assert!(bvh.nodes[0].right_child < bvh.nodes.len() as u32);
    }
}
