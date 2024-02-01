use bevy::asset::{load_internal_asset, Asset};
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::mesh::MeshVertexBufferLayout;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderRef, SpecializedMeshPipelineError,
};
// use bevy::render::RenderApp;
// use bevy::render::renderer::{RenderAdapter, RenderDevice};

pub const CURVATURE_MESH_VERTEX_OUTPUT: Handle<Shader> = Handle::weak_from_u128(128741983741982);

pub const UTIL: Handle<Shader> = Handle::weak_from_u128(128742342344982);

pub const MAX_VERTICES: usize = 200000;
pub const MAX_TRIANGLES: usize = 2 * MAX_VERTICES - 5;

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone, TypeUuid)]
#[uuid = "b62bb455-a72c-4b56-87bb-81e0554e234f"]
pub struct TerrainMaterial {
    #[uniform(0)]
    pub max_height: f32,
    #[uniform(1)]
    pub grass_line: f32,
    #[uniform(2)]
    pub tree_line: f32,
    #[uniform(3)]
    pub snow_line: f32,
    #[uniform(4)]
    pub grass_color: Color,
    #[uniform(5)]
    pub tree_color: Color,
    #[uniform(6)]
    pub snow_color: Color,
    #[uniform(7)]
    pub stone_color: Color,
    #[uniform(8)]
    pub cosine_max_snow_slope: f32,
    #[uniform(9)]
    pub cosine_max_tree_slope: f32,
    // This is the linear map bounds when trying to convert to our texture space
    // it's hard to explain, sorry
    #[uniform(10)]
    pub u_bound: f32,
    #[uniform(11)]
    pub v_bound: f32,
    #[texture(12)]
    #[sampler(13)]
    pub(crate) normal_texture: Option<Handle<Image>>,
    #[uniform(14)]
    pub(crate) scale: f32,

    /// Interpolation related stuff!
    /// len is MAX_TRIANGLES * 3
    #[storage(15, read_only)]
    pub(crate) triangles: Vec<u32>,
    /// len is MAX_TRIANGLES * 3
    #[storage(16, read_only)]
    pub(crate) halfedges: Vec<u32>,
    /// len is 2 * MAX_VERTICES
    #[storage(17, read_only)]
    pub(crate) vertices: Vec<f32>,
    /// len is MAX_VERTICES
    #[storage(18, read_only)]
    pub(crate) height: Vec<f32>,
    /// len is 2 * MAX_VERTICES
    #[storage(19, read_only)]
    pub(crate) gradients: Vec<f32>,
    // We manually add this later, since it's a storage texture and those aren't supported yet
    /// len is TERRAIN_VERTICES
    #[storage(20)]
    pub(crate) triangle_indices: Vec<u32>,
}

/// The Material trait is very configurable, but comes with sensible defaults for all methods.
/// You only need to implement functions for features that need non-default behavior. See the Material api docs for details!
impl Material for TerrainMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/curvature_transform.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/terrain_color.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayout,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        return Ok(());
    }
}

pub struct TerrainPlugin {}

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            CURVATURE_MESH_VERTEX_OUTPUT,
            "../assets/shaders/curvature_mesh_vertex_output.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(app, UTIL, "../assets/shaders/util.wgsl", Shader::from_wgsl);
    }
}