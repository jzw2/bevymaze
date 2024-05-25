use std::array;
use std::sync::Arc;

use bevy::asset::Assets;
use bevy::ecs::reflect::ReflectComponent;
use bevy::math::{Vec2, Vec3};
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::{
    default, Commands, Component, Deref, Entity, Query, Reflect, Res, ResMut, Resource, Transform,
    TransformBundle, With,
};
use bevy::reflect::Array;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy_rapier3d::prelude::Collider;
use bevy_tokio_tasks::TokioTasksRuntime;
use bitvec::vec::BitVec;
use crossbeam_channel::{bounded, Receiver, Sender};
use futures_lite::StreamExt;
use futures_util::SinkExt;
use image::{ImageBuffer, Rgb, RgbImage};
use itertools::Itertools;
use postcard::{from_bytes, to_stdvec};
use tokio::task::JoinHandle;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message::Binary;
use url::Url;

use crate::loader::ServerRequestSender;
use server::connection::{CompressedMazeComponent, MazeNetworkRequest, MazeNetworkResponse};
use server::square_maze_gen::{SQUARE_MAZE_CELL_COUNT, SQUARE_MAZE_CELL_SIZE};

use crate::maze_loader::MazeComponentDataState::{Invalid, Loading, Valid};
use crate::maze_render::MAZE_HEIGHT;
use crate::shaders::{MazeLayerMaterial, MazeLayerMaterialDataHolder, TerrainMaterial};
use crate::terrain_loader::{MainTerrainColldier, MeshOffsetRes};

pub struct MazeDataUpdate {
    data: RawMazeData,
    maze_top_left: Vec2,
    colliders: Vec<(Collider, TransformBundle, MazeColldier)>,
}

#[derive(Resource, Deref)]
pub struct MazeResponseSender(pub Sender<Vec<CompressedMazeComponent>>);

#[derive(Resource, Deref)]
pub struct MazeResponseReceiver(pub Receiver<Vec<CompressedMazeComponent>>);

#[derive(Resource, Deref)]
pub struct MazeUpdateReceiver(pub Receiver<MazeDataUpdate>);

#[derive(Resource)]
pub struct MazeUpdateProcHandle(pub JoinHandle<()>);

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct MazeColldier;

pub fn setup_maze_loader(
    mut commands: Commands,
    runtime: ResMut<TokioTasksRuntime>,
    transform: Res<MeshOffsetRes>,
    maze_data_holder: Res<MazeDataHolder>,
    server_request_sender: ResMut<ServerRequestSender>,
    server_response_receiver: ResMut<MazeResponseReceiver>,
) {
    let (tx, rx) = bounded::<MazeDataUpdate>(5);
    let transform = Arc::clone(&transform.0);

    let sender = server_request_sender.0.clone();
    let receiver = server_response_receiver.0.clone();
    // we just use this once, and it lives ~forever~ (spooky)
    let mut holder = maze_data_holder.clone();
    // let mesh_verts = terrain_data_map.mesh.terrain_mesh_verts.clone();
    commands.insert_resource::<MazeUpdateProcHandle>(MazeUpdateProcHandle(
        runtime.spawn_background_task(|_ctx| async move {
            loop {
                // update the center
                {
                    let t = transform.lock().unwrap();
                    // TODO: update center of maze holder
                    holder.top_left = [
                        -(MAZE_COMPONENTS_X as i32) / 2,
                        -(MAZE_COMPONENTS_Y as i32) / 2,
                    ]
                    // holder.mesh.center = t.translation().xz();
                    // println!("Trans {:?}", map.center);
                }

                let batch_size = 1024; //512 / size_of::<TerrainDataPoint>();
                let mut to_fetch = Vec::<[i32; 2]>::with_capacity(batch_size);
                let mut total = 0;

                let [left, top] = holder.top_left;
                for i in 0..MAZE_COMPONENTS_X {
                    for j in 0..MAZE_COMPONENTS_Y {
                        match holder.data_states[i][j] {
                            Invalid => {
                                to_fetch.push([i as i32 + left, j as i32 + top]);
                                holder.data_states[i][j] = Loading;
                            }
                            _ => {}
                        }

                        // send off a load and mark requested
                        // we do this so the server can start fetching data for us
                        if to_fetch.len() >= batch_size
                            || (i == MAZE_COMPONENTS_X - 1
                                && j == MAZE_COMPONENTS_Y - 1
                                && to_fetch.len() > 0)
                        {
                            total += to_fetch.len();
                            sender
                                .send(MazeNetworkRequest::ReqMaze(to_fetch.clone()))
                                .expect("TODO: panic message");
                            // now clear and restart
                            to_fetch.clear();
                        }
                    }
                }

                if total == 0 {
                    continue;
                }

                while total > 0 {
                    if let Some(fetched) = receiver.iter().next() {
                        for cell in &fetched {
                            // println!("Importing {:?} with data [{}{}{}{}{}{}...]", cell.cell, cell.data[0], cell.data[1], cell.data[2], cell.data[3], cell.data[4], cell.data[5]);
                            holder.import(cell);
                        }
                        total -= fetched.len();
                    }
                }

                tx.send(MazeDataUpdate {
                    data: holder.raw_data.clone(),
                    maze_top_left: Vec2::new(holder.top_left[0] as f32, holder.top_left[1] as f32),
                    colliders: generate_maze_colliders(&holder),
                })
                .unwrap();
            }
        }),
    ));

    commands.insert_resource(MazeUpdateReceiver(rx));
}

enum WallDirection {
    Horizontal,
    Vertical,
}
type WallExistenceMat =
    [[bool; SQUARE_MAZE_CELL_COUNT as usize * 2]; SQUARE_MAZE_CELL_COUNT as usize * 2];

#[derive(Debug, Clone, Copy)]
struct BestSegment {
    idx: usize,
    length: usize,
    segment: [usize; 2],
}
fn get_best_segment(matrix: &WallExistenceMat, direction: WallDirection) -> BestSegment {
    let mut best_col_segment = BestSegment {
        idx: usize::MAX,
        length: 0,
        segment: [0, 0],
    };
    let get_existence = |i: usize, j: usize| -> bool {
        match direction {
            WallDirection::Horizontal => matrix[j][i],
            WallDirection::Vertical => matrix[i][j],
        }
    };
    // Now find the longest wall strip vertically
    for i in 0..matrix.len() / 2 {
        let mut best_segment = [0, 0];
        let mut best_length = 0;
        let mut cur_segment = [0, 0];
        let mut cur_length = 0;
        for j in 0..matrix.len() {
            let existence = get_existence(2 * i, j);
            if existence {
                cur_length += 1;
                cur_segment = [cur_segment[0], j + 1]
            }

            if cur_length > best_length {
                best_segment = cur_segment;
                best_length = cur_length;
            }

            if !existence {
                cur_length = 0;
                cur_segment = [j + 1, j + 1];
            }
        }
        if best_length > best_col_segment.length {
            best_col_segment = BestSegment {
                idx: 2 * i,
                length: best_length,
                segment: best_segment,
            }
        }
    }
    return best_col_segment;
}

pub fn generate_maze_colliders(
    maze_data: &MazeDataHolder,
) -> Vec<(Collider, TransformBundle, MazeColldier)> {
    // first generate the wall-existence matrix
    let mut matrix =
        [[false; SQUARE_MAZE_CELL_COUNT as usize * 2]; SQUARE_MAZE_CELL_COUNT as usize * 2];
    let mut wall_count = 0;

    const CELL_COUNT: usize = SQUARE_MAZE_CELL_COUNT as usize;
    for x in 4..5 {
        for y in 4..5 {
            for i in 0..CELL_COUNT {
                for j in 0..CELL_COUNT {
                    // top left always a wall
                    matrix[2 * i][2 * j] = true;
                    wall_count += 1;
                    // left path
                    let pos = 2 * (i + j * SQUARE_MAZE_CELL_COUNT as usize);
                    matrix[2 * i][2 * j + 1] = maze_data.data[x][y][pos];
                    wall_count += maze_data.data[x][y][pos] as usize;
                    // right path
                    matrix[2 * i + 1][2 * j] = maze_data.data[x][y][pos + 1];
                    wall_count += maze_data.data[x][y][pos + 1] as usize;
                }
            }
        }
    }

    let mut colliders = vec![];

    while wall_count > 0 {
        let best_vert = get_best_segment(&matrix, WallDirection::Vertical);
        let best_horiz = get_best_segment(&matrix, WallDirection::Horizontal);

        let top_left;
        let bottom_right;

        println!(
            "Best walls {:?} {:?} Wall count {wall_count}",
            best_vert, best_horiz
        );
        let vec: String = matrix
            .into_iter()
            .map(|c| c.into_iter().map(|e| format!("{}", e as u8)).join(" "))
            .join("\n");
        println!("Matrix\n{vec}");

        if best_vert.length > best_horiz.length {
            for s in best_vert.segment[0]..best_vert.segment[1] {
                matrix[best_vert.idx][s] = false;
            }
            // now create the collider
            let top_square = [best_vert.segment[0], best_vert.idx];
            let bottom_square = [best_vert.segment[1], best_vert.idx + 1];
            // TODO: fill in height
            top_left = Vec3::new(
                top_square[0] as f32 * SQUARE_MAZE_CELL_SIZE as f32,
                0.,
                top_square[1] as f32 * SQUARE_MAZE_CELL_SIZE as f32,
            );
            bottom_right = Vec3::new(
                bottom_square[0] as f32 * SQUARE_MAZE_CELL_SIZE as f32,
                0.,
                bottom_square[1] as f32 * SQUARE_MAZE_CELL_SIZE as f32,
            );

            wall_count -= best_vert.length;
        } else {
            for s in best_horiz.segment[0]..best_horiz.segment[1] {
                matrix[s][best_horiz.idx] = false;
            }
            // now create the collider
            let left_square = [best_vert.idx, best_horiz.segment[0]];
            let right_square = [best_vert.idx + 1, best_horiz.segment[0]];
            // TODO: fill in height
            top_left = Vec3::new(
                left_square[0] as f32 * SQUARE_MAZE_CELL_SIZE as f32,
                0.,
                left_square[1] as f32 * SQUARE_MAZE_CELL_SIZE as f32,
            );
            bottom_right = Vec3::new(
                right_square[0] as f32 * SQUARE_MAZE_CELL_SIZE as f32,
                0.,
                right_square[1] as f32 * SQUARE_MAZE_CELL_SIZE as f32,
            );

            wall_count -= best_horiz.length;
        }

        let dims = (bottom_right - top_left);
        colliders.push((
            Collider::cuboid(dims.x / 2., MAZE_HEIGHT * 10., dims.z / 2.),
            TransformBundle {
                local: Transform::from_translation((top_left + bottom_right) / 2.0),
                ..default()
            },
            MazeColldier,
        ));
    }

    return colliders;
}

pub fn stream_maze_mesh(
    mut commands: Commands,
    maze_colldier: Query<Entity, With<MazeColldier>>,
    receiver: Res<MazeUpdateReceiver>,
    mut maze_material: ResMut<Assets<ExtendedMaterial<TerrainMaterial, MazeLayerMaterial>>>,
    mut terrain_material_data_holder: ResMut<MazeLayerMaterialDataHolder>,
    mut render_device: ResMut<RenderDevice>,
    mut render_queue: ResMut<RenderQueue>,
) {
    let last = receiver.try_iter().last();
    if let Some(update) = last {
        maze_colldier.iter().for_each(|c| {
            commands.entity(c).despawn();
        });
        for collider_bundle in update.colliders {
            commands.spawn(collider_bundle);
        }
        // const WIDTH: u32 = (MAZE_COMPONENTS_X as i64 * SQUARE_MAZE_CELL_COUNT * 2) as u32;
        // const HEIGHT: u32 = (MAZE_COMPONENTS_Y as i64 * SQUARE_MAZE_CELL_COUNT * 2) as u32;
        // let mut image: RgbImage = ImageBuffer::from_pixel(WIDTH, HEIGHT, Rgb([0, 0, 0]));
        //
        // for i in 0..WIDTH / 2 {
        //     for j in 0..HEIGHT / 2 {
        //         *image.get_pixel_mut(i * 2, j * 2) = Rgb([255, 255, 255]);
        //     }
        // }
        //
        // for i in 0..MAZE_COMPONENTS_X as i64 {
        //     for j in 0..MAZE_COMPONENTS_Y as i64 {
        //         let bits = BitVec::<u32>::from_vec(Vec::from(update.data[i as usize][j as usize]));
        //         for s in 0..SQUARE_MAZE_CELL_COUNT {
        //             for t in 0..SQUARE_MAZE_CELL_COUNT {
        //                 // check for the left edge
        //                 let cpx = 2 * (i * SQUARE_MAZE_CELL_COUNT + s);
        //                 let cpy = 2 * (j * SQUARE_MAZE_CELL_COUNT + t);
        //                 let pos = 2 * (s + t * SQUARE_MAZE_CELL_COUNT) as usize;
        //
        //                 if bits[pos] {
        //                     let lp = (cpx, cpy + 1);
        //                     *image.get_pixel_mut(lp.0 as u32, lp.1 as u32) = Rgb([255, 255, 255]);
        //                 }
        //
        //                 if bits[pos + 1] {
        //                     let tp = (cpx + 1, cpy);
        //                     *image.get_pixel_mut(tp.0 as u32, tp.1 as u32) = Rgb([255, 255, 255]);
        //                 }
        //             }
        //         }
        //     }
        // }
        //
        // image.save("maze_loader_output.png").unwrap();

        terrain_material_data_holder.raw_maze_data.set(update.data);
        terrain_material_data_holder
            .raw_maze_data
            .write_buffer(&*render_device, &*render_queue);

        if let Some((handle, material)) = maze_material.iter_mut().next() {
            material.extension.maze_top_left = update.maze_top_left;
        }
    }
}

pub(crate) const MAZE_COMPONENTS_X: usize = 8;
pub(crate) const MAZE_COMPONENTS_Y: usize = MAZE_COMPONENTS_X;
pub(crate) const MAZE_DATA_COUNT: usize =
    (2 * SQUARE_MAZE_CELL_COUNT * SQUARE_MAZE_CELL_COUNT / 32) as usize;

pub type MazeData = Box<[[BitVec<u32>; MAZE_COMPONENTS_Y]; MAZE_COMPONENTS_X]>;
pub type RawMazeData = Box<[[[u32; MAZE_DATA_COUNT]; MAZE_COMPONENTS_Y]; MAZE_COMPONENTS_X]>;

#[derive(Clone, Copy)]
pub enum MazeComponentDataState {
    Valid,
    Invalid,
    Loading,
}

#[derive(Resource, Clone)]
pub struct MazeDataHolder {
    top_left: [i32; 2],
    buffer_start: [usize; 2],
    data_states: [[MazeComponentDataState; MAZE_COMPONENTS_Y]; MAZE_COMPONENTS_X],
    data: MazeData,
    raw_data: RawMazeData,
}

impl MazeDataHolder {
    pub fn new() -> Self {
        return MazeDataHolder {
            top_left: [0, 0],
            buffer_start: [0, 0],
            data_states: [[Invalid; MAZE_COMPONENTS_Y]; MAZE_COMPONENTS_X],
            data: Box::new(array::from_fn(|i| array::from_fn(|j| BitVec::<u32>::new()))),
            raw_data: Box::new([[[0u32; MAZE_DATA_COUNT]; MAZE_COMPONENTS_Y]; MAZE_COMPONENTS_X]),
        };
    }

    /// This function updates the offset
    /// We treat `data` sort of like a 2d ring buffer.
    /// We never shift the physical location of the maze data around.
    /// Instead, we change the location of the beginning of the buffer and let it wrap around.
    /// So if we go right 2 cells, then the top left will be shifted left two cells. Everything behind
    /// the top left will be marked as invalid until it's fetched. The two entries left of `top_left`
    /// become
    pub fn set_offset(&mut self, new_off: [i32; 2]) {
        let [new_x, new_y] = new_off;
        let [old_x, old_y] = self.top_left;
        let [x_change, y_change] = [new_x - old_x, new_y - old_y];
        // mark all the cells behind or in front of our new x-offset as invalid
        if x_change > 0 {
            for x in 0..x_change.min(MAZE_COMPONENTS_X as i32) {
                for y in 0..MAZE_COMPONENTS_Y {
                    self.data_states[(old_x + x) as usize % MAZE_COMPONENTS_X][y] = Invalid;
                }
            }
        } else if x_change < 0 {
            for x in x_change.max(-(MAZE_COMPONENTS_X as i32))..0 {
                for y in 0..MAZE_COMPONENTS_Y {
                    self.data_states[(old_x + x) as usize % MAZE_COMPONENTS_X][y] = Invalid;
                }
            }
        }

        if y_change > 0 {
            for y in 0..y_change.min(MAZE_COMPONENTS_Y as i32) {
                for x in 0..MAZE_COMPONENTS_X {
                    self.data_states[x][(old_y + y) as usize % MAZE_COMPONENTS_Y] = Invalid;
                }
            }
        } else if y_change < 0 {
            for y in y_change.max(-(MAZE_COMPONENTS_Y as i32))..0 {
                for x in 0..MAZE_COMPONENTS_X {
                    self.data_states[x][(old_y + y) as usize % MAZE_COMPONENTS_Y] = Invalid;
                }
            }
        }
    }

    pub fn import(&mut self, component: &CompressedMazeComponent) {
        let [x, y] = component.position;
        let [top, left] = self.top_left;
        let [i, j] = [x - left, y - top];
        if i < 0 || j < 0 || i as usize > MAZE_COMPONENTS_X || j as usize > MAZE_COMPONENTS_Y {
            return;
        }
        let [i, j] = [
            (i as usize + self.buffer_start[0]) % MAZE_COMPONENTS_X,
            (j as usize + self.buffer_start[1]) % MAZE_COMPONENTS_Y,
        ];
        self.data[i][j] = component.data.clone();
        self.raw_data[i][j] = component.data.clone().into_vec().try_into().unwrap();
        self.data_states[i][j] = Valid;
    }
}
