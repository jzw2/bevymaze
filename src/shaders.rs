use bevy::asset::{
    embedded_asset, load_internal_asset, Asset, UntypedAssetId, VisitAssetDependencies,
};
use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::core_pipeline::core_3d;
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::ecs::query::QueryItem;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::{TypePath, TypeUuid};
use bevy::render::extract_component::{
    ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
};
use bevy::render::mesh::MeshVertexBufferLayout;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraph, RenderGraphApp, RenderGraphContext,
};
use bevy::render::render_resource::{
    AsBindGroup, AsBindGroupError, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, CachedComputePipelineId, CachedPipelineState,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, ComputePassDescriptor,
    ComputePipelineDescriptor, FragmentState, MultisampleState, Operations, PipelineCache,
    PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
    Sampler, SamplerBindingType, SamplerDescriptor, ShaderRef, ShaderStages, ShaderType,
    SpecializedMeshPipelineError, TextureFormat, TextureSampleType, TextureViewDimension,
    UnpreparedBindGroup,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::{BevyDefault, FallbackImage};
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderApp};
use std::borrow::Cow;

pub const CURVATURE_MESH_VERTEX_OUTPUT: Handle<Shader> = Handle::weak_from_u128(128741983741982);

pub const UTIL: Handle<Shader> = Handle::weak_from_u128(128742342344982);

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

        // we also need to add our mesh interpolation triangle-finding faze to the pipeline
        app.add_plugins((
            // The settings will be a component that lives in the main world but will
            // be extracted to the render world every frame.
            // This makes it possible to control the effect from the main world.
            // This plugin will take care of extracting it automatically.
            // It's important to derive [`ExtractComponent`] on [`PostProcessingSettings`]
            // for this plugin to work correctly.
            ExtractComponentPlugin::<RawTerrainTriangulationData>::default(),
            // The settings will also be the data used in the shader.
            // This plugin will prepare the component for the GPU by creating a uniform buffer
            // and writing the data to that buffer every frame.
            UniformComponentPlugin::<RawTerrainTriangulationData>::default(),
        ));

        let render_app = app.sub_app_mut(RenderApp);
        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();

        render_graph.add_node(core_3d::graph::NAME, TerrainInterpolationNode::default());
        render_app.add_render_graph_edges(
            core_3d::graph::NAME,
            &[
                TerrainInterpolationNode::NAME,
                core_3d::graph::node::PREPASS,
            ],
        );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<TerrainInterpolationPipeline>();
    }
}

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
        layout: &MeshVertexBufferLayout,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

// The post process node used for the render graph
#[derive(Default)]
struct TerrainInterpolationNode;
impl TerrainInterpolationNode {
    pub const NAME: &'static str = "post_process";
}

// The ViewNode trait is required by the ViewNodeRunner
impl Node for TerrainInterpolationNode {
    // Runs the node logic
    // This is where you encode draw commands.
    //
    // This will run on every view on which the graph is running.
    // If you don't want your effect to run on every camera,
    // you'll need to make sure you have a marker component as part of [`ViewQuery`]
    // to identify which camera(s) should run the effect.
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // The pipeline cache is a cache of all previously created pipelines.
        // It is required to avoid creating a new pipeline each frame,
        // which is expensive due to shader compilation.
        let pipeline_cache = world.resource::<PipelineCache>();

        // Get the pipeline from the cache
        let pipeline = world.resource::<TerrainInterpolationPipeline>();

        // Get the settings uniform binding
        let triangulation_uniforms =
            world.resource::<ComponentUniforms<RawTerrainTriangulationData>>();
        let Some(triangulation_bindings) = triangulation_uniforms.uniforms().binding() else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        let pipeline = pipeline_cache
            .get_compute_pipeline(pipeline.pipeline)
            .unwrap();
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(100, 1, 1);

        Ok(())
    }
}

// This contains global data used by the render pipeline. This will be created once on startup.
#[derive(Resource)]
struct TerrainInterpolationPipeline {
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for TerrainInterpolationPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // We need to define the bind group layout used for our pipeline
        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("terrain_interpolation_bind_group_layout"),
            entries: &[
                // The buffer containing the triangulation
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: bevy::render::render_resource::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: Some(RawTerrainTriangulationData::min_size()),
                    },
                    count: None,
                },
            ],
        });

        // Get the shader handle
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/terrain_interpolation.wgsl");

        let pipeline_cache = world.resource_mut::<PipelineCache>();

        let pipeline = pipeline_cache
            // This will add the pipeline to the cache and queue it's creation
            .queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("terrain_interpolation_pipeline".into()),
                layout: vec![layout.clone()],
                push_constant_ranges: vec![],
                shader,
                shader_defs: vec![],
                // TODO: determine what this means. init is probably incorrect, IDK enough yet
                entry_point: Cow::from("init"),
            });

        Self { layout, pipeline }
    }
}

// This is the component that will get passed to the shader
#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
struct RawTerrainTriangulationData {
    intensity: f32,
    // WebGL2 structs must be 16 byte aligned.
    #[cfg(feature = "webgl2")]
    _webgl2_padding: Vec3,
}
