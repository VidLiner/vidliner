//! Parallel kernels for VidLiner's video data path.
//!
//! This crate deliberately contains no filesystem, model, or runtime-profile code. It is the
//! candidate native accelerator for the pure operations that dominate large video datasets:
//! projecting many frame labels and finding one-axis contrast pairs. Rayon is used for parallel
//! work, while indexed collection and a final sort preserve deterministic output order.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Box2D {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Geometry {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub tx: f64,
    pub ty: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Placement {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Transform {
    pub geometry: Geometry,
    pub placement: Placement,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct TimeTransform {
    pub scale: f64,
    pub offset: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Span {
    pub t0: f64,
    pub t1: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContrastPair {
    pub left: usize,
    pub right: usize,
    pub axis: String,
}

fn point(g: Geometry, x: f64, y: f64) -> (f64, f64) {
    (g.a * x + g.c * y + g.tx, g.b * x + g.d * y + g.ty)
}

/// Project one normalised box through an affine transform and output placement.
pub fn project_box(transform: Transform, input: Box2D) -> Box2D {
    let points = [
        point(transform.geometry, input.x_min, input.y_min),
        point(transform.geometry, input.x_max, input.y_min),
        point(transform.geometry, input.x_min, input.y_max),
        point(transform.geometry, input.x_max, input.y_max),
    ];
    let x_min = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let y_min = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let x_max = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_max = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    Box2D {
        x_min: transform.placement.x + x_min * transform.placement.w,
        y_min: transform.placement.y + y_min * transform.placement.h,
        x_max: transform.placement.x + x_max * transform.placement.w,
        y_max: transform.placement.y + y_max * transform.placement.h,
    }
}

/// Project independent frame boxes in parallel while retaining input order.
pub fn project_boxes_parallel(transform: Transform, boxes: &[Box2D]) -> Vec<Box2D> {
    boxes.par_iter().map(|input| project_box(transform, *input)).collect()
}

/// Project independent time spans in parallel and preserve positive ordering.
pub fn project_spans_parallel(transform: TimeTransform, spans: &[Span]) -> Vec<Span> {
    spans
        .par_iter()
        .map(|span| {
            let a = transform.scale * span.t0 + transform.offset;
            let b = transform.scale * span.t1 + transform.offset;
            Span { t0: a.min(b), t1: a.max(b) }
        })
        .collect()
}

/// Find all pairs whose fingerprints differ on exactly one axis.
///
/// The outer index is parallelised. Sorting after collection makes the result independent of the
/// worker count, which is essential for reproducible manifests and cache keys.
pub fn contrast_pairs(fingerprints: &[BTreeMap<String, String>]) -> Vec<ContrastPair> {
    let mut pairs: Vec<ContrastPair> = (0..fingerprints.len())
        .into_par_iter()
        .flat_map_iter(|left| {
            (left + 1..fingerprints.len()).filter_map(move |right| {
                let axes: BTreeSet<&String> = fingerprints[left]
                    .keys()
                    .chain(fingerprints[right].keys())
                    .collect();
                let differing: Vec<&String> = axes
                    .into_iter()
                    .filter(|axis| fingerprints[left].get(*axis) != fingerprints[right].get(*axis))
                    .collect();
                if differing.len() == 1 {
                    Some(ContrastPair { left, right, axis: differing[0].clone() })
                } else {
                    None
                }
            })
        })
        .collect();
    pairs.sort_by(|a, b| (a.left, a.right, &a.axis).cmp(&(b.left, b.right, &b.axis)));
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Transform {
        Transform {
            geometry: Geometry { a: 1.0, b: 0.0, c: 0.0, d: 1.0, tx: 0.0, ty: 0.0 },
            placement: Placement { x: 0.0, y: 0.0, w: 1.0, h: 1.0 },
        }
    }

    #[test]
    fn parallel_projection_preserves_order() {
        let boxes = vec![
            Box2D { x_min: 0.1, y_min: 0.2, x_max: 0.3, y_max: 0.4 },
            Box2D { x_min: 0.5, y_min: 0.6, x_max: 0.7, y_max: 0.8 },
        ];
        assert_eq!(project_boxes_parallel(identity(), &boxes), boxes);
    }

    #[test]
    fn spans_are_normalised_after_mapping() {
        let spans = project_spans_parallel(TimeTransform { scale: 2.0, offset: -1.0 }, &[Span { t0: 0.5, t1: 1.0 }]);
        assert_eq!(spans, vec![Span { t0: 0.0, t1: 1.0 }]);
    }

    #[test]
    fn contrast_pairs_are_deterministic() {
        let mut first = BTreeMap::new();
        first.insert("format".to_owned(), "source".to_owned());
        let mut second = first.clone();
        second.insert("format".to_owned(), "square".to_owned());
        let mut third = first.clone();
        third.insert("style".to_owned(), "warm".to_owned());
        assert_eq!(contrast_pairs(&[first, second, third]).len(), 2);
    }
}
