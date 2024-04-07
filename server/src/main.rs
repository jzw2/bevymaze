use futures_util::{SinkExt, StreamExt};
use log::*;
use postcard::{from_bytes, to_stdvec};
use server::connection::{MazeCell, MazeNetworkRequest, MazeNetworkResponse, TerrainDataPoint};
use server::maze_gen::CompressedMaze;
use server::square_maze_gen::{load_or_generate_component, SQUARE_MAZE_CELL_COUNT};
use server::terrain_gen::TerrainGenerator;
use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message::Binary;
use tokio_tungstenite::{
    accept_async,
    tungstenite::{Error, Result},
};

async fn accept_connection(peer: SocketAddr, stream: TcpStream) {
    if let Err(e) = handle_connection(peer, stream).await {
        match e {
            Error::ConnectionClosed | Error::Protocol(_) | Error::Utf8 => (),
            err => error!("Error processing connection: {}", err),
        }
    }
}

async fn handle_connection(peer: SocketAddr, stream: TcpStream) -> Result<()> {
    let mut ws_stream = accept_async(stream).await.expect("Failed to accept");

    let gen = TerrainGenerator::new();

    info!("New WebSocket connection: {}", peer);

    while let Some(msg) = ws_stream.next().await {
        let msg = msg?;
        if let Binary(bin) = msg {
            // println!("Intercepted message!");
            let mut to_get = from_bytes::<MazeNetworkRequest>(&bin).unwrap();
            match to_get {
                MazeNetworkRequest::ReqTerrainHeights(mut data) => {
                    println!("RTH {}", data.len());
                    for point in &mut data {
                        // println!("BEFORE {:?} {:?} {} {}", point.coordinates, point.gradient, point.height, point.idx);
                        let [x, y] = point.coordinates;
                        point.height = gen.get_height_for(x as f64, y as f64) as f32;
                        point.gradient = gen.get_gradient(x as f64, y as f64).as_vec2().to_array();
                        // println!("AFTER {:?} {:?} {} {}", point.coordinates, point.gradient, point.height, point.idx);
                    }
                    ws_stream
                        .send(Binary(
                            to_stdvec(&MazeNetworkResponse::TerrainHeights(data)).unwrap(),
                        ))
                        .await?;
                }
                MazeNetworkRequest::ReqMaze(data) => {
                    println!("RM {}", data.len());
                    let filled: Vec<MazeCell> = data
                        .iter()
                        .map(|cell| {
                            let [x, y] = cell;
                            return MazeCell {
                                cell: [*x, *y],
                                data: load_or_generate_component((*x as i64, *y as i64))
                                    .compressed(),
                            };
                        })
                        .collect();
                    ws_stream
                        .send(Binary(
                            to_stdvec(&MazeNetworkResponse::Maze(filled)).unwrap(),
                        ))
                        .await?;
                }
            }
            // println!("Fetching {}", to_get.len());
        }
    }

    Ok(())
}

use image::{ImageBuffer, Rgb, RgbImage};

const CELLS: i64 = 4;

const WIDTH: u32 = (CELLS * SQUARE_MAZE_CELL_COUNT * 2) as u32;
const HEIGHT: u32 = WIDTH;

#[tokio::main]
async fn main() {
    env_logger::init();

    // let mut image: RgbImage = ImageBuffer::from_pixel(WIDTH, HEIGHT, Rgb([0, 0, 0]));

    // for i in 0..WIDTH / 2 {
    //     for j in 0..HEIGHT / 2 {
    //         *image.get_pixel_mut(i * 2, j * 2) = Rgb([255, 255, 255]);
    //     }
    // }

    // for i in 0..CELLS {
    //     for j in 0..CELLS {
    //         let comp = load_or_generate_component((i, j));
    //         let (x, y) = comp.offset();
    //         for s in 0..comp.size {
    //             for t in 0..comp.size {
    //                 // check for the left edge
    //                 let cpx = 2 * (i * comp.size + s);
    //                 let cpy = 2 * (j * comp.size + t);
    //                 let node = (x + s, y + t);
    //                 if comp.maze.contains_edge((x + s - 1, y + t), node) {
    //                     let lp = (cpx, cpy + 1);
    //                     *image.get_pixel_mut(lp.0 as u32, lp.1 as u32) = Rgb([255, 255, 255]);
    //                 }

    //                 if comp.maze.contains_edge((x + s, y + t - 1), node) {
    //                     let tp = (cpx + 1, cpy);
    //                     *image.get_pixel_mut(tp.0 as u32, tp.1 as u32) = Rgb([255, 255, 255]);
    //                 }
    //             }
    //         }

    //         // for ((n1x, n1y), (n2x, n2y), _) in comp.maze.all_edges() {
    //         //     let (i1x, i1y) = (
    //         //         n1x - x + i * SQUARE_MAZE_CELL_COUNT,
    //         //         n1y - y + j * SQUARE_MAZE_CELL_COUNT,
    //         //     );
    //         //     let (i2x, i2y) = (
    //         //         n2x - x + i * SQUARE_MAZE_CELL_COUNT,
    //         //         n2y - y + j * SQUARE_MAZE_CELL_COUNT,
    //         //     );

    //         //     let mut p: (i64, i64);

    //         //     if i1x == i2x {
    //         //         if i1y > i2y {
    //         //             p = (i1x * 2 + 1, i1y * 2);
    //         //         } else {
    //         //             p = (i2x * 2 + 1, i2y * 2);
    //         //         }
    //         //     } else if i1y == i2y {
    //         //         if i1x > i2x {
    //         //             p = (i1x * 2, i1y * 2 + 1);
    //         //         } else {
    //         //             p = (i2x * 2, i2y * 2 + 1);
    //         //         }
    //         //     } else {
    //         //         panic!("Invalid edge")
    //         //     }

    //         //     println!(
    //         //         "{i} {j} | {} {} | {x} {y} | {n1x} {n1y} {n2x} {n2y} | {} {} {} {} | {i1x} {i1y} {i2x} {i2y} | {} {}",
    //         //         i - CELLS / 2,
    //         //         j - CELLS / 2,
    //         //         n1x - x,
    //         //         n1y - y,
    //         //         n2x - x,
    //         //         n2y - y,
    //         //         p.0,
    //         //         p.1
    //         //     );

    //         //     if p.0 < 0 || p.1 < 0 || p.0 as u32 >= WIDTH || p.1 as u32 >= HEIGHT {
    //         //         continue;
    //         //     }

    //         //     *image.get_pixel_mut(p.0 as u32, p.1 as u32) = Rgb([255, 255, 255]);
    //         // }
    //     }
    // }

    // image.save("output.png").unwrap();

    let addr = "127.0.0.1:9002";
    let listener = TcpListener::bind(&addr).await.expect("Can't listen");
    info!("Listening on: {}", addr);
    
    while let Ok((stream, _)) = listener.accept().await {
        let peer = stream
            .peer_addr()
            .expect("connected streams should have a peer address");
        info!("Peer address: {}", peer);
    
        tokio::spawn(accept_connection(peer, stream));
    }
}
