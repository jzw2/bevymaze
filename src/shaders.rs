use bevy::asset::{load_internal_asset, Asset};
use bevy::core_pipeline::core_3d;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypeUuid;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::mesh::MeshVertexBufferLayout;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{RenderGraph, RenderGraphApp, ViewNodeRunner};
use bevy::render::render_resource::{
    encase, AsBindGroup, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBindingType, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
    CachedPipelineState, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
    RenderPipelineDescriptor, ShaderRef, ShaderStages, ShaderType, SpecializedMeshPipelineError,
    StorageTextureAccess, TextureFormat, TextureViewDimension,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use bevy::render::{render_graph, Render, RenderApp, RenderSet};
use bevy_mod_debugdump::render_graph::Settings;
use std::borrow::Cow;
use std::collections::HashMap;
// use bevy::render::RenderApp;
// use bevy::render::renderer::{RenderAdapter, RenderDevice};

/// Lots of stuff copied from https://github.com/mwbryant/logic_compute_shaders

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

        /*
        Begin particle_system.rs
         */
        app.add_plugins(ExtractComponentPlugin::<ParticleSystem>::default());

        let render_app = app.sub_app_mut(RenderApp);

        // let update_node = UpdateParticlesNode::new(&mut render_app.world);
        // let render_node = RenderParticlesNode::new(&mut render_app.world);
        //
        // let mut render_graph = render_app.world.resource_mut::<RenderGraph>();

        // render_graph.add_node("update_particles", update_node);
        // render_graph.add_node("render_particles", render_node);
        //
        // render_graph.add_node_edge("update_particles", "render_particles");
        // render_graph.add_node_edge(
        //     "render_particles",
        //     bevy::render::main_graph::node::CAMERA_DRIVER,
        // );

        render_app
            .add_render_graph_node::<UpdateTerrainVertexHeightsNode>(
                core_3d::graph::NAME,
                UPDATE_TERRAIN_VERTEX_HEIGHTS_NODE_NAME,
            )
            .add_render_graph_node::<RenderParticlesNode>(
                core_3d::graph::NAME,
                RENDER_PARTICLES_NODE_NAME,
            )
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[UPDATE_TERRAIN_VERTEX_HEIGHTS_NODE_NAME, RENDER_PARTICLES_NODE_NAME],
            )
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    RENDER_PARTICLES_NODE_NAME,
                    core_3d::graph::node::START_MAIN_PASS,
                ],
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<ParticleUpdatePipeline>()
            .init_resource::<ParticleSystemRender>()
            .init_resource::<ParticleRenderPipeline>();
        //     .add_systems(Render, queue_bind_group.in_set(RenderSet::Queue));

        let dot = bevy_mod_debugdump::render_graph_dot(
            &app,
            &Settings {
                style: Default::default(),
            },
        );

        std::fs::write("render-graph.dot", dot).expect("Failed to write render-graph.dot");
    }
    /*
    End particle_system.rs
     */
}

/*
Begin section copied from main.rs
 */

pub const HEIGHT: f32 = 480.0;
pub const WIDTH: f32 = 640.0;

pub const PARTICLE_COUNT: u32 = 1000;
// XXX when changing this also change it in the shader... TODO figure out how to avoid that...
pub const WORKGROUP_SIZE: u32 = 16;

#[derive(ShaderType, Default, Clone, Copy)]
struct Particle {
    position: Vec2,
}

#[derive(Component, Default, Clone)]
pub struct ParticleSystem {
    pub rendered_texture: Handle<Image>,
}

/*
End main.rs
 */

/*
Begin section copied from compute_utils.rs
*/

pub fn compute_pipeline_descriptor(
    shader: Handle<Shader>,
    entry_point: &str,
    bind_group_layout: &BindGroupLayout,
) -> ComputePipelineDescriptor {
    ComputePipelineDescriptor {
        label: None,
        layout: vec![bind_group_layout.clone()],
        push_constant_ranges: vec![],
        shader,
        shader_defs: vec![],
        entry_point: Cow::from(entry_point.to_owned()),
    }
}

pub fn run_compute_pass(
    render_context: &mut RenderContext,
    bind_group: &BindGroup,
    pipeline_cache: &PipelineCache,
    pipeline: CachedComputePipelineId,
) {
    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());

    pass.set_bind_group(0, bind_group, &[]);

    let pipeline = pipeline_cache.get_compute_pipeline(pipeline).unwrap();
    pass.set_pipeline(pipeline);

    pass.dispatch_workgroups(PARTICLE_COUNT / WORKGROUP_SIZE, 1, 1);
}

//ugh lazy dupe
pub fn run_compute_pass_2d(
    render_context: &mut RenderContext,
    bind_group: &BindGroup,
    pipeline_cache: &PipelineCache,
    pipeline: CachedComputePipelineId,
) {
    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor::default());

    pass.set_bind_group(0, bind_group, &[]);

    let pipeline = pipeline_cache.get_compute_pipeline(pipeline).unwrap();
    pass.set_pipeline(pipeline);

    pass.dispatch_workgroups(
        WIDTH as u32 / WORKGROUP_SIZE,
        HEIGHT as u32 / WORKGROUP_SIZE,
        1,
    );
}

/*
End compute_utils.rs
 */

/*
Begin particle_system.rs
 */
// Must maintain all our own data because render world flushes between frames :,(
#[derive(Resource, Default)]
pub struct ParticleSystemRender {
    // pub update_bind_group: bevy::utils::HashMap<Entity, BindGroup>,
    // pub render_bind_group: bevy::utils::HashMap<Entity, BindGroup>,
    // pub particle_buffers: bevy::utils::HashMap<Entity, Buffer>,
}

// no need to queue the bind group since we're just using the bindings from the material

// fn queue_bind_group(
//     render_device: Res<RenderDevice>,
//     //render_queue: Res<RenderQueue>,
//     render_pipeline: Res<ParticleRenderPipeline>,
//     gpu_images: Res<RenderAssets<Image>>,
//     mut particle_system_render: ResMut<ParticleSystemRender>,
//     update_pipeline: Res<ParticleUpdatePipeline>,
//     //Getting mutable queries in the render world is an antipattern?
//     particle_systems: Query<(Entity, &ParticleSystem)>,
// ) {
//     // Everything here is done lazily and should only happen on the first call here.
//     for (entity, system) in &particle_systems {
//         if !particle_system_render
//             .particle_buffers
//             .contains_key(&entity)
//         {
//             let particle = [Particle::default(); PARTICLE_COUNT as usize];
//             //ugh
//             let mut byte_buffer = Vec::new();
//             let mut buffer = encase::StorageBuffer::new(&mut byte_buffer);
//             buffer.write(&particle).unwrap();
//
//             let storage = render_device.create_buffer_with_data(&BufferInitDescriptor {
//                 label: None,
//                 usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
//                 contents: buffer.into_inner(),
//             });
//
//             particle_system_render
//                 .particle_buffers
//                 .insert(entity, storage);
//         }
//
//         /*
//         read_buffer(
//             &particle_systems_render.particle_buffers[&entity],
//             &render_device,
//             &render_queue,
//         );
//         */
//
//         if !particle_system_render
//             .update_bind_group
//             .contains_key(&entity)
//         {
//             let update_group = update_bind_group(
//                 entity,
//                 &render_device,
//                 &update_pipeline,
//                 &particle_system_render,
//             );
//             particle_system_render
//                 .update_bind_group
//                 .insert(entity, update_group);
//         }
//
//         if !particle_system_render
//             .render_bind_group
//             .contains_key(&entity)
//         {
//             let view = gpu_images.get(&system.rendered_texture).unwrap();
//             let render_group = render_bind_group(
//                 entity,
//                 &render_device,
//                 &render_pipeline,
//                 &particle_system_render,
//                 view,
//             );
//
//             particle_system_render
//                 .render_bind_group
//                 .insert(entity, render_group);
//         }
//     }
// }

impl ExtractComponent for ParticleSystem {
    type Query = &'static ParticleSystem;
    type Filter = ();
    type Out = Self;

    fn extract_component(item: bevy::ecs::query::QueryItem<'_, Self::Query>) -> Option<Self> {
        Some(item.clone())
    }
}

/*
End particle_system.rs
 */

/*
Begin particle_update.rs
 */
#[derive(Resource, Clone)]
pub struct ParticleUpdatePipeline {
    bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

pub const UPDATE_TERRAIN_VERTEX_HEIGHTS_NODE_NAME: &str = "update_terrain_vertex_heights";
pub struct UpdateTerrainVertexHeightsNode {
    // particle_systems: QueryState<Entity, With<ParticleSystem>>,
    // update_state: bevy::utils::HashMap<Entity, ParticleUpdateState>,
}

#[derive(Default, Clone)]
enum ParticleUpdateState {
    #[default]
    Loading,
    Init,
    Update,
}

fn update_bind_group_layout() -> BindGroupLayoutDescriptor<'static> {
    BindGroupLayoutDescriptor {
        label: None,
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    }
}

// pub fn update_bind_group(
//     entity: Entity,
//     render_device: &RenderDevice,
//     update_pipeline: &ParticleUpdatePipeline,
//     particle_system_render: &ParticleSystemRender,
// ) -> BindGroup {
//     let entries = [BindGroupEntry {
//         binding: 0,
//         resource: BindingResource::Buffer(
//             particle_system_render.particle_buffers[&entity].as_entire_buffer_binding(),
//         ),
//     }];
//     render_device.create_bind_group(None, &update_pipeline.bind_group_layout, &entries)
// }

impl FromWorld for ParticleUpdatePipeline {
    fn from_world(world: &mut World) -> Self {
        // let bind_group_layout = world
        //     .resource::<RenderDevice>()
        //     .create_bind_group_layout(&update_bind_group_layout());

        let bind_group_layout =
            TerrainMaterial::bind_group_layout(world.resource::<RenderDevice>());

        let shader = world.resource::<AssetServer>().load("particle_update.wgsl");

        let mut pipeline_cache = world.resource_mut::<PipelineCache>();

        let init_pipeline = pipeline_cache.queue_compute_pipeline(compute_pipeline_descriptor(
            shader.clone(),
            "init",
            &bind_group_layout,
        ));

        let update_pipeline = pipeline_cache.queue_compute_pipeline(compute_pipeline_descriptor(
            shader,
            "update",
            &bind_group_layout,
        ));

        ParticleUpdatePipeline {
            bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

impl render_graph::Node for UpdateTerrainVertexHeightsNode {
    fn update(&mut self, world: &mut World) {
        // nothing?

        // let mut systems = world.query_filtered::<Entity, With<ParticleSystem>>();
        // let pipeline = world.resource::<ParticleUpdatePipeline>();
        // let pipeline_cache = world.resource::<PipelineCache>();
        //
        // for entity in systems.iter(world) {
        //     // if the corresponding pipeline has loaded, transition to the next stage
        //     self.update_state(entity, pipeline_cache, pipeline);
        // }
        // //Update the query for the run step
        // self.particle_systems.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ParticleUpdatePipeline>();
        // let particle_systems_render = world.resource::<ParticleSystemRender>();

        run_compute_pass(
            render_context,
            // &particle_systems_render.update_bind_group[&entity],
            pipeline_cache,
            pipeline,
        );

        // for entity in self.particle_systems.iter_manual(world) {
        //     // select the pipeline based on the current state
        //     if let Some(pipeline) = match self.update_state[&entity] {
        //         ParticleUpdateState::Loading => None,
        //         ParticleUpdateState::Init => Some(pipeline.init_pipeline),
        //         ParticleUpdateState::Update => Some(pipeline.update_pipeline),
        //     } {
        //         run_compute_pass(
        //             render_context,
        //             &particle_systems_render.update_bind_group[&entity],
        //             pipeline_cache,
        //             pipeline,
        //         );
        //     }
        // }

        Ok(())
    }
}

impl UpdateTerrainVertexHeightsNode {
    pub fn new(world: &mut World) -> Self {
        Self {
            // particle_systems: QueryState::new(world),
            // update_state: bevy::utils::HashMap::default(),
        }
    }

    fn update_state(
        &mut self,
        entity: Entity,
        pipeline_cache: &PipelineCache,
        pipeline: &ParticleUpdatePipeline,
    ) {
        let update_state = match self.update_state.get(&entity) {
            Some(state) => state,
            None => {
                self.update_state
                    .insert(entity, ParticleUpdateState::Loading);
                &ParticleUpdateState::Loading
            }
        };

        match update_state {
            ParticleUpdateState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline)
                {
                    self.update_state.insert(entity, ParticleUpdateState::Init);
                }
            }
            ParticleUpdateState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.update_state
                        .insert(entity, ParticleUpdateState::Update);
                }
            }
            ParticleUpdateState::Update => {}
        }
    }
}

/*
End particle_update.rs
 */

impl FromWorld for UpdateTerrainVertexHeightsNode {
    fn from_world(world: &mut World) -> Self {
        return UpdateTerrainVertexHeightsNode::new(world);
    }
}

/*
Begin particle_render.rs
 */
#[derive(Resource, Clone)]
pub struct ParticleRenderPipeline {
    bind_group_layout: BindGroupLayout,
    clear_pipeline: CachedComputePipelineId,
    render_pipeline: CachedComputePipelineId,
}

pub const RENDER_PARTICLES_NODE_NAME: &str = "render_particles";

pub struct RenderParticlesNode {
    particle_systems: QueryState<Entity, With<ParticleSystem>>,
    render_state: bevy::utils::HashMap<Entity, ParticleRenderState>,
}

#[derive(Default, Clone, Debug)]
enum ParticleRenderState {
    #[default]
    Loading,
    Render,
}

// fn bind_group_layout() -> BindGroupLayoutDescriptor<'static> {
//     BindGroupLayoutDescriptor {
//         label: None,
//         entries: &[
//             BindGroupLayoutEntry {
//                 binding: 0,
//                 visibility: ShaderStages::COMPUTE,
//                 ty: BindingType::Buffer {
//                     ty: BufferBindingType::Storage { read_only: false },
//                     has_dynamic_offset: false,
//                     min_binding_size: None,
//                 },
//                 count: None,
//             },
//             BindGroupLayoutEntry {
//                 binding: 1,
//                 visibility: ShaderStages::COMPUTE,
//                 ty: BindingType::StorageTexture {
//                     access: StorageTextureAccess::ReadWrite,
//                     format: TextureFormat::Rgba8Unorm,
//                     view_dimension: TextureViewDimension::D2,
//                 },
//                 count: None,
//             },
//         ],
//     }
// }

// pub fn render_bind_group(
//     entity: Entity,
//     render_device: &RenderDevice,
//     render_pipeline: &ParticleRenderPipeline,
//     particle_system_render: &ParticleSystemRender,
//     view: &GpuImage,
// ) -> BindGroup {
//     let entries = [
//         BindGroupEntry {
//             binding: 0,
//             resource: BindingResource::Buffer(
//                 particle_system_render.particle_buffers[&entity].as_entire_buffer_binding(),
//             ),
//         },
//         BindGroupEntry {
//             binding: 1,
//             resource: BindingResource::TextureView(&view.texture_view),
//         },
//     ];
//
//     render_device.create_bind_group(None, &render_pipeline.bind_group_layout, &entries)
// }

impl FromWorld for ParticleRenderPipeline {
    fn from_world(world: &mut World) -> Self {
        // let bind_group_layout = world
        //     .resource::<RenderDevice>()
        //     .create_bind_group_layout(&bind_group_layout());
        let bind_group_layout =
            TerrainMaterial::bind_group_layout(world.resource::<RenderDevice>());

        let shader = world.resource::<AssetServer>().load("particle_render.wgsl");
        let mut pipeline_cache = world.resource_mut::<PipelineCache>();

        let render_pipeline = pipeline_cache.queue_compute_pipeline(compute_pipeline_descriptor(
            shader.clone(),
            "render",
            &bind_group_layout,
        ));

        let clear_pipeline = pipeline_cache.queue_compute_pipeline(compute_pipeline_descriptor(
            shader,
            "clear",
            &bind_group_layout,
        ));

        ParticleRenderPipeline {
            bind_group_layout,
            clear_pipeline,
            render_pipeline,
        }
    }
}

impl render_graph::Node for RenderParticlesNode {
    fn update(&mut self, world: &mut World) {
        let mut systems = world.query_filtered::<Entity, With<ParticleSystem>>();
        let pipeline = world.resource::<ParticleRenderPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        for entity in systems.iter(world) {
            self.update_state(entity, pipeline_cache, pipeline);
        }

        self.particle_systems.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ParticleRenderPipeline>();
        let particle_systems_render = world.resource::<ParticleSystemRender>();

        for entity in self.particle_systems.iter_manual(world) {
            if let Some((clear_pipeline, render_pipeline)) = match self.render_state[&entity] {
                ParticleRenderState::Loading => None,
                ParticleRenderState::Render => {
                    Some((pipeline.clear_pipeline, pipeline.render_pipeline))
                }
            } {
                // run_compute_pass_2d(
                //     render_context,
                //     &particle_systems_render.render_bind_group[&entity],
                //     pipeline_cache,
                //     clear_pipeline,
                // );
                run_compute_pass(
                    render_context,
                    &particle_systems_render.render_bind_group[&entity],
                    pipeline_cache,
                    render_pipeline,
                );
            }
        }

        Ok(())
    }
}

impl RenderParticlesNode {
    pub fn new(world: &mut World) -> Self {
        Self {
            particle_systems: QueryState::new(world),
            render_state: bevy::utils::HashMap::default(),
        }
    }
    fn update_state(
        &mut self,
        entity: Entity,
        pipeline_cache: &PipelineCache,
        pipeline: &ParticleRenderPipeline,
    ) {
        let render_state = match self.render_state.get(&entity) {
            Some(state) => state,
            None => {
                self.render_state
                    .insert(entity, ParticleRenderState::Loading);
                &ParticleRenderState::Loading
            }
        };
        match render_state {
            ParticleRenderState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.render_pipeline)
                {
                    self.render_state
                        .insert(entity, ParticleRenderState::Render);
                }
            }
            ParticleRenderState::Render => {}
        }
    }
}

/*
End particle_render.rs
 */

impl FromWorld for RenderParticlesNode {
    fn from_world(world: &mut World) -> Self {
        return RenderParticlesNode::new(world);
    }
}
