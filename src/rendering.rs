use std::cmp::Reverse;
use std::ops::Range;
use bevy::prelude::{Entity, Query, Res};
use bevy::render::render_phase::{CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions, PhaseItem};
use bevy::render::render_resource::CachedRenderPipelineId;
use bevy::render::view::{ExtractedView, VisibleEntities};
use bevy::math::FloatOrd;

// pub struct TerrainOpaque3d {
//     /// the priority, assigned by God
//     pub priority: f32,
//     
//     pub pipeline: CachedRenderPipelineId,
//     pub entity: Entity,
//     pub draw_function: DrawFunctionId,
//     pub batch_range: Range<u32>,
//     pub dynamic_offset: Option<u32>,
// }
// 
// impl PhaseItem for TerrainOpaque3d {
//     // NOTE: Values increase towards the camera. Front-to-back ordering for opaque means we need a descending sort.
//     type SortKey = FloatOrd;
// 
//     #[inline]
//     fn entity(&self) -> Entity {
//         self.entity
//     }
// 
//     #[inline]
//     fn sort_key(&self) -> Self::SortKey {
//         FloatOrd(self.priority)
//     }
// 
//     #[inline]
//     fn draw_function(&self) -> DrawFunctionId {
//         self.draw_function
//     }
// 
//     #[inline]
//     fn sort(items: &mut [Self]) {
//         // Key negated to match reversed SortKey ordering
//         items.sort_by_key(|item| item.sort_key());
//         // radsort::sort_by_key(items, |item| -item.distance);
//     }
// 
//     #[inline]
//     fn batch_range(&self) -> &Range<u32> {
//         &self.batch_range
//     }
// 
//     #[inline]
//     fn batch_range_mut(&mut self) -> &mut Range<u32> {
//         &mut self.batch_range
//     }
// 
//     #[inline]
//     fn dynamic_offset(&self) -> Option<NonMaxU32> {
//         self.dynamic_offset
//     }
// 
//     #[inline]
//     fn dynamic_offset_mut(&mut self) -> &mut Option<NonMaxU32> {
//         &mut self.dynamic_offset
//     }
// }
// 
// impl CachedRenderPipelinePhaseItem for TerrainOpaque3d {
//     #[inline]
//     fn cached_pipeline(&self) -> CachedRenderPipelineId {
//         self.pipeline
//     }
// }

// pub(crate) fn queue_terrain(
//     // cuboids_pipelines: Res<CuboidsPipelines>,
//     opaque_3d_draw_functions: Res<DrawFunctions<TerrainOpaque3d>>,
//     // buffer_cache: Res<CuboidBufferCache>,
//     mut views: Query<(
//         &ExtractedView,
//         &VisibleEntities,
//         &mut RenderPhase<TerrainOpaque3d>,
//     )>,
// ) {
//     let draw_cuboids = opaque_3d_draw_functions
//         .read()
//         .get_id::<DrawCuboids>()
//         .unwrap();
// 
//     for (view, visible_entities, mut opaque_phase) in views.iter_mut() {
//         // TODO: add method so we can use this on a vector
//         // let range_finder = view.rangefinder3d();
//         let inverse_view_matrix = view.transform.compute_matrix().inverse();
//         let inverse_view_row_2 = inverse_view_matrix.row(2);
// 
//         for &entity in &visible_entities.entities {
//             if let Some(entry) = buffer_cache.entries.get(&entity) {
//                 if entry.enabled {
//                     let pipeline = if view.hdr {
//                         cuboids_pipelines.hdr_pipeline_id
//                     } else {
//                         cuboids_pipelines.pipeline_id
//                     };
//                     opaque_phase.add(AabbOpaque3d {
//                         distance: inverse_view_row_2.dot(entry.position.extend(1.0)),
//                         pipeline,
//                         entity,
//                         draw_function: draw_cuboids,
//                         batch_range: 0..1,
//                         dynamic_offset: None,
//                     });
//                 }
//             }
//         }
//     }
// }