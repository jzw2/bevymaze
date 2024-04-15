use std::sync::Arc;

use bevy::asset::Assets;
use bevy::math::Vec2;
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::{Commands, Deref, Res, ResMut, Resource};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy_tokio_tasks::TokioTasksRuntime;
use bitvec::vec::BitVec;
use crossbeam_channel::{bounded, Receiver};
use futures_lite::StreamExt;
use futures_util::SinkExt;
use image::{ImageBuffer, Rgb, RgbImage};
use postcard::{from_bytes, to_stdvec};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message::Binary;
use url::Url;

use server::connection::{MazeCell, MazeNetworkRequest, MazeNetworkResponse};
use server::square_maze_gen::SQUARE_MAZE_CELL_COUNT;

use crate::maze_loader::MazeCellDataState::{Invalid, Loading, Valid};
use crate::shaders::{MazeLayerMaterial, MazeLayerMaterialDataHolder, TerrainMaterial};
use crate::terrain_loader::MeshOffsetRes;

pub struct MazeDataUpdate {
    data: MazeData,
    maze_top_left: Vec2,
}

#[derive(Resource, Deref)]
pub struct MazeUpdateReceiver(Receiver<MazeDataUpdate>);

pub fn setup_maze_loader(
    mut commands: Commands,
    runtime: ResMut<TokioTasksRuntime>,
    transform: Res<MeshOffsetRes>,
    maze_data_holder: Res<MazeDataHolder>,
    // mut terrain_material: ResMut<Assets<TerrainMaterial>>,
    // mut terrain_material_data_holder: ResMut<TerrainMaterialDataHolder>,
    mut render_device: ResMut<RenderDevice>,
    mut render_queue: ResMut<RenderQueue>,
) {
    let (tx, rx) = bounded::<MazeDataUpdate>(5);
    let transform = Arc::clone(&transform.0);

    // we just use this once, and it lives ~forever~ (spooky)
    let mut holder = maze_data_holder.clone();
    // let mesh_verts = terrain_data_map.mesh.terrain_mesh_verts.clone();
    runtime.spawn_background_task(|_ctx| async move {
        // now fetch from the server!
        let (mut socket, other) =
            connect_async(Url::parse("ws://127.0.0.1:9002").expect("Can't connect to URL"))
                .await
                .unwrap();

        loop {
            // update the center
            {
                let t = transform.lock().unwrap();
                // TODO: update center of maze holder
                holder.top_left = [-(MAZE_CELLS_X as i32) / 2, -(MAZE_CELLS_Y as i32) / 2]
                // holder.mesh.center = t.translation().xz();
                // println!("Trans {:?}", map.center);
            }

            let batch_size = 1024; //512 / size_of::<TerrainDataPoint>();
            let mut to_fetch = Vec::<[i32; 2]>::with_capacity(batch_size);
            let mut total = 0;

            let [left, top] = holder.top_left;
            for i in 0..MAZE_CELLS_X {
                for j in 0..MAZE_CELLS_Y {
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
                        || (i == MAZE_CELLS_X - 1 && j == MAZE_CELLS_Y - 1 && to_fetch.len() > 0)
                    {
                        total += to_fetch.len();
                        socket
                            .send(Binary(
                                to_stdvec(&MazeNetworkRequest::ReqMaze(to_fetch.clone())).unwrap(),
                            ))
                            .await
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
                match socket.next().await.unwrap() {
                    Ok(msg) => {
                        if let Binary(bin) = msg {
                            let fetched = from_bytes::<MazeNetworkResponse>(&bin).unwrap();
                            if let MazeNetworkResponse::Maze(fetched) = fetched {
                                for cell in &fetched {
                                    // println!("Importing {:?} with data [{}{}{}{}{}{}...]", cell.cell, cell.data[0], cell.data[1], cell.data[2], cell.data[3], cell.data[4], cell.data[5]);
                                    holder.import(cell);
                                }
                                total -= fetched.len();
                            }
                        }
                    }
                    Err(_) => {
                        // do nothing
                    }
                }
            }

            tx.send(MazeDataUpdate {
                data: holder.data.clone(),
                maze_top_left: Vec2::new(holder.top_left[0] as f32, holder.top_left[1] as f32),
            })
            .unwrap();
        }
    });
    // tokio::spawn();

    commands.insert_resource(MazeUpdateReceiver(rx));
}

pub fn stream_maze_mesh(
    mut commands: Commands,
    receiver: Res<MazeUpdateReceiver>,
    mut maze_material: ResMut<Assets<ExtendedMaterial<TerrainMaterial, MazeLayerMaterial>>>,
    mut terrain_material_data_holder: ResMut<MazeLayerMaterialDataHolder>,
    mut render_device: ResMut<RenderDevice>,
    mut render_queue: ResMut<RenderQueue>,
) {
    let last = receiver.try_iter().last();
    if let Some(update) = last {
        println!("Setting maze data");
        const WIDTH: u32 = (MAZE_CELLS_X as i64 * SQUARE_MAZE_CELL_COUNT * 2) as u32;
        const HEIGHT: u32 = (MAZE_CELLS_Y as i64 * SQUARE_MAZE_CELL_COUNT * 2) as u32;
        let mut image: RgbImage = ImageBuffer::from_pixel(WIDTH, HEIGHT, Rgb([0, 0, 0]));

        for i in 0..WIDTH / 2 {
            for j in 0..HEIGHT / 2 {
                *image.get_pixel_mut(i * 2, j * 2) = Rgb([255, 255, 255]);
            }
        }

        for i in 0..MAZE_CELLS_X as i64 {
            for j in 0..MAZE_CELLS_Y as i64 {
                let bits = BitVec::<u32>::from_vec(Vec::from(update.data[i as usize][j as usize]));
                for s in 0..SQUARE_MAZE_CELL_COUNT {
                    for t in 0..SQUARE_MAZE_CELL_COUNT {
                        // check for the left edge
                        let cpx = 2 * (i * SQUARE_MAZE_CELL_COUNT + s);
                        let cpy = 2 * (j * SQUARE_MAZE_CELL_COUNT + t);
                        let pos = 2 * (s + t * SQUARE_MAZE_CELL_COUNT) as usize;

                        if bits[pos] {
                            let lp = (cpx, cpy + 1);
                            *image.get_pixel_mut(lp.0 as u32, lp.1 as u32) = Rgb([255, 255, 255]);
                        }

                        if bits[pos + 1] {
                            let tp = (cpx + 1, cpy);
                            *image.get_pixel_mut(tp.0 as u32, tp.1 as u32) = Rgb([255, 255, 255]);
                        }
                    }
                }
            }
        }

        image.save("maze_loader_output.png").unwrap();

        terrain_material_data_holder.raw_maze_data.set(update.data);
        terrain_material_data_holder
            .raw_maze_data
            .write_buffer(&*render_device, &*render_queue);

        if let Some((handle, material)) = maze_material.iter_mut().next() {
            material.extension.maze_top_left = update.maze_top_left;
        }
    }
}

pub(crate) const MAZE_CELLS_X: usize = 8;
pub(crate) const MAZE_CELLS_Y: usize = MAZE_CELLS_X;
pub(crate) const MAZE_DATA_COUNT: usize =
    (2 * SQUARE_MAZE_CELL_COUNT * SQUARE_MAZE_CELL_COUNT / 32) as usize;

pub type MazeData = Box<[[[u32; MAZE_DATA_COUNT]; MAZE_CELLS_Y]; MAZE_CELLS_X]>;

#[derive(Clone, Copy)]
pub enum MazeCellDataState {
    Valid,
    Invalid,
    Loading,
}

#[derive(Resource, Clone)]
pub struct MazeDataHolder {
    top_left: [i32; 2],
    buffer_start: [usize; 2],
    data_states: [[MazeCellDataState; MAZE_CELLS_Y]; MAZE_CELLS_X],
    data: MazeData,
}

impl MazeDataHolder {
    pub fn new() -> Self {
        return MazeDataHolder {
            top_left: [0, 0],
            buffer_start: [0, 0],
            data_states: [[Invalid; MAZE_CELLS_Y]; MAZE_CELLS_X],
            data: Box::new([[[0u32; MAZE_DATA_COUNT]; MAZE_CELLS_Y]; MAZE_CELLS_X]),
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
            for x in 0..x_change.min(MAZE_CELLS_X as i32) {
                for y in 0..MAZE_CELLS_Y {
                    self.data_states[(old_x + x) as usize % MAZE_CELLS_X][y] = Invalid;
                }
            }
        } else if x_change < 0 {
            for x in x_change.max(-(MAZE_CELLS_X as i32))..0 {
                for y in 0..MAZE_CELLS_Y {
                    self.data_states[(old_x + x) as usize % MAZE_CELLS_X][y] = Invalid;
                }
            }
        }

        if y_change > 0 {
            for y in 0..y_change.min(MAZE_CELLS_Y as i32) {
                for x in 0..MAZE_CELLS_X {
                    self.data_states[x][(old_y + y) as usize % MAZE_CELLS_Y] = Invalid;
                }
            }
        } else if y_change < 0 {
            for y in y_change.max(-(MAZE_CELLS_Y as i32))..0 {
                for x in 0..MAZE_CELLS_X {
                    self.data_states[x][(old_y + y) as usize % MAZE_CELLS_Y] = Invalid;
                }
            }
        }
    }

    pub fn import(&mut self, cell: &MazeCell) {
        let [x, y] = cell.cell;
        let [top, left] = self.top_left;
        let [i, j] = [x - left, y - top];
        if i < 0 || j < 0 || i as usize > MAZE_CELLS_X || j as usize > MAZE_CELLS_Y {
            return;
        }
        let [i, j] = [
            (i as usize + self.buffer_start[0]) % MAZE_CELLS_X,
            (j as usize + self.buffer_start[1]) % MAZE_CELLS_Y,
        ];
        self.data[i][j] = cell.data.clone().into_vec().try_into().unwrap();
        self.data_states[i][j] = Valid;
    }
}
