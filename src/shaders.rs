use crate::terrain_render::TERRAIN_VERTICES;
use bevy::asset::{load_internal_asset, Asset};
use bevy::core_pipeline::core_3d;
use bevy::core_pipeline::core_3d::graph::node::START_MAIN_PASS;
use bevy::pbr::{ExtendedMaterial, ExtractedMaterials, MaterialExtension, MaterialPipeline, MaterialPipelineKey, PrepassPipeline, RenderMaterials};
use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_instances::ExtractedInstances;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::mesh::MeshVertexBufferLayout;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{RenderGraph, RenderGraphApp, ViewNodeRunner};
use bevy::render::render_resource::{encase, AsBindGroup, AsBindGroupError, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, BufferInitDescriptor, BufferUsages, BufferVec, CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor, ComputePipelineDescriptor, OwnedBindingResource, PipelineCache, PipelineCacheError, PreparedBindGroup, RenderPipelineDescriptor, ShaderRef, ShaderStages, ShaderType, SpecializedMeshPipelineError, StorageTextureAccess, TextureFormat, TextureViewDimension, UnpreparedBindGroup, StorageBuffer};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::{FallbackImage, GpuImage};
use bevy::render::{render_graph, Render, RenderApp, RenderSet};
use bevy_mod_debugdump::render_graph::Settings;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;
use std::sync::Arc;
use crate::maze_loader::MazeData;
// use bevy::render::RenderApp;
// use bevy::render::renderer::{RenderAdapter, RenderDevice};

/// Lots of stuff copied from https://github.com/mwbryant/logic_compute_shaders

pub const CURVATURE_MESH_VERTEX_OUTPUT: Handle<Shader> = Handle::weak_from_u128(128741983741982);

pub const UTIL: Handle<Shader> = Handle::weak_from_u128(128742342344982);

pub const MAX_VERTICES: usize = 200000;
pub const MAX_TRIANGLES: usize = 2 * MAX_VERTICES - 5;

#[derive(Resource)]
pub struct TerrainMaterialDataHolder {
    /// len is MAX_TRIANGLES * 3
    pub(crate) triangles: BufferVec<u32>,
    /// len is MAX_TRIANGLES * 3
    pub(crate) halfedges: BufferVec<u32>,
    /// len is 2 * MAX_VERTICES
    pub(crate) vertices: BufferVec<f32>,
    /// len is MAX_VERTICES
    pub(crate) height: BufferVec<f32>,
    /// len is 2 * MAX_VERTICES
    pub(crate) gradients: BufferVec<f32>,
    // We manually add this later, since it's a storage texture and those aren't supported yet
    /// len is TERRAIN_VERTICES
    pub(crate) triangle_indices: BufferVec<u32>,

    // the vertices of the mesh on the x-z plane
    // len is TERRAIN_VERTICES * 2
    pub(crate) mesh_vertices: BufferVec<f32>,
    // TODO: implement transform the same way!
    // // The transform applied to the mesh every frame
    // #[storage(22, read_only, visibility(vertex, fragment, compute))]
    // pub(crate) transform: Vec2,
}

impl TerrainMaterialDataHolder {
    pub(crate) fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.vertices.write_buffer(device, queue);
        self.triangles.write_buffer(device, queue);
        self.halfedges.write_buffer(device, queue);
        self.height.write_buffer(device, queue);
        self.gradients.write_buffer(device, queue);
        self.triangle_indices.write_buffer(device, queue);
    }
}

#[derive(Asset, TypePath, TypeUuid, AsBindGroup, Debug, Clone)]
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
    #[storage(15, read_only, buffer, visibility(vertex, fragment, compute))]
    pub(crate) triangles: Buffer,
    /// len is MAX_TRIANGLES * 3
    #[storage(16, read_only, buffer, visibility(vertex, fragment, compute))]
    pub(crate) halfedges: Buffer,
    /// len is 2 * MAX_VERTICES
    #[storage(17, read_only, buffer, visibility(vertex, fragment, compute))]
    pub(crate) vertices: Buffer,
    /// len is MAX_VERTICES
    #[storage(18, read_only, buffer, visibility(vertex, fragment, compute))]
    pub(crate) height: Buffer,
    /// len is 2 * MAX_VERTICES
    #[storage(19, read_only, buffer, visibility(vertex, fragment, compute))]
    pub(crate) gradients: Buffer,
    // We manually add this later, since it's a storage texture and those aren't supported yet
    /// len is TERRAIN_VERTICES
    #[storage(20, buffer, visibility(vertex, fragment, compute))]
    pub(crate) triangle_indices: Buffer,

    // the vertices of the mesh on the x-z plane
    // len is TERRAIN_VERTICES * 2
    #[storage(21, read_only, buffer, visibility(vertex, fragment, compute))]
    pub(crate) mesh_vertices: Buffer,
    // The transform applied to the mesh every frame
    #[storage(22, read_only, visibility(vertex, fragment, compute))]
    pub(crate) transform: Vec2,
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

#[derive(Resource)]
pub struct MazeLayerMaterialDataHolder {
    pub(crate) raw_maze_data: StorageBuffer<MazeData>
}

#[derive(Asset, TypePath, TypeUuid, AsBindGroup, Debug, Clone)]
#[uuid = "b62bb455-a72c-4b56-87bb-81e0554e234b"]
pub struct MazeLayerMaterial {
    #[uniform(23)]
    pub layer_height: f32,

    #[storage(24, read_only, buffer, visibility(vertex, fragment, compute))]
    pub(crate) data: Buffer,
    
    #[uniform(25)]
    pub(crate) maze_top_left: Vec2,
}

impl MaterialExtension for MazeLayerMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/maze_curvature_transform.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/maze_color.wgsl".into()
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

        app.add_plugins(MaterialPlugin::<TerrainMaterial> { ..default() });
        app.add_plugins(MaterialPlugin::<ExtendedMaterial<TerrainMaterial, MazeLayerMaterial>> { ..default() });
        
        let render_app = app.sub_app_mut(RenderApp);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();

        render_app
            .add_render_graph_node::<UpdateTerrainVertexHeightsNode>(
                core_3d::graph::NAME,
                UPDATE_TERRAIN_VERTEX_HEIGHTS_NODE_NAME,
            )
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[UPDATE_TERRAIN_VERTEX_HEIGHTS_NODE_NAME, START_MAIN_PASS],
            );

        
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<UpdateTerrainHeightsPipeline>();
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
struct GameOfLifeLabel;

#[derive(Resource)]
struct UpdateTerrainHeightsPipeline {
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for UpdateTerrainHeightsPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let terrain_bind_group_layout = TerrainMaterial::bind_group_layout(render_device);
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/terrain_heights.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![terrain_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("init"),
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![terrain_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        UpdateTerrainHeightsPipeline {
            init_pipeline,
            update_pipeline,
        }
    }
}

enum UpdateTerrainVertexHeightsState {
    Loading,
    Init,
    Update,
}

const UPDATE_TERRAIN_VERTEX_HEIGHTS_NODE_NAME: &str = "update_terrain_heights";

struct UpdateTerrainVertexHeightsNode {
    state: UpdateTerrainVertexHeightsState,
}

impl Default for UpdateTerrainVertexHeightsNode {
    fn default() -> Self {
        Self {
            state: UpdateTerrainVertexHeightsState::Loading,
        }
    }
}

impl render_graph::Node for UpdateTerrainVertexHeightsNode {
    fn update(&mut self, world: &mut World) {
        // println!("Run update");
        let pipeline = world.resource::<UpdateTerrainHeightsPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            UpdateTerrainVertexHeightsState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
                {
                    self.state = UpdateTerrainVertexHeightsState::Init;
                } else if let CachedPipelineState::Err(err) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
                {
                    match err {
                        PipelineCacheError::ShaderNotLoaded(id) => {
                            println!("SNL")
                        }
                        PipelineCacheError::ProcessShaderError(err) => {
                            println!("PSE")
                        }
                        PipelineCacheError::ShaderImportNotYetAvailable => {
                            println!("SINYA")
                        }
                        PipelineCacheError::CreateShaderModule(err) => {
                            println!("CSM {}", err)
                        }
                    }
                }
            }
            UpdateTerrainVertexHeightsState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = UpdateTerrainVertexHeightsState::Update;
                }
            }
            UpdateTerrainVertexHeightsState::Update => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let render_terrain_material = world.resource::<RenderMaterials<TerrainMaterial>>();
        let (id, prepared_material) = render_terrain_material.0.iter().next().unwrap();

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<UpdateTerrainHeightsPipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, &prepared_material.bind_group, &[]);

        // select the pipeline based on the current state
        match self.state {
            UpdateTerrainVertexHeightsState::Loading => {
                // println!("load");
            }
            UpdateTerrainVertexHeightsState::Init => {
                // println!("init");
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(600, 1, 1);
            }
            UpdateTerrainVertexHeightsState::Update => {
                // println!("Update");
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(8, 1, 1);
            }
        }

        Ok(())
    }
}
