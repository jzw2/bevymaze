use byteorder::{NativeEndian, ReadBytesExt, WriteBytesExt};
use futures_util::{SinkExt, StreamExt};
use log::*;
use regex::Regex;
use server::terrain_data::{compressed_height, idx_to_coords, DATUM_COUNT, TILE_DIM};
use server::terrain_gen::{TerrainGenerator, TILE_SIZE};
use server::util::lin_map;
use std::fs::{read, File};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::net::SocketAddr;
use std::path::Path;
use std::str::FromStr;
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

/// Copied from Stackoverflow
/// https://stackoverflow.com/a/29042896/3210986
fn as_u8_slice<T>(v: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<T>())
    }
}

async fn get_from_coordinates(peer: SocketAddr, stream: TcpStream) -> Result<()> {
     let mut ws_stream = accept_async(stream).await.expect("Failed to accept");

    info!("New WebSocket connection: {}", peer);

    while let Some(msg) = ws_stream.next().await {
        let msg = msg?;
        if msg.is_binary() {
            let bin = msg.into_data()?;
        }
        
        let mut buffer: Vec<u16>; 
        if msg.is_text() {
            let text = msg.into_text()?;
            let reg = Regex::new(r"([0-9]+),([0-9]+):([0-9]+)").unwrap();
            let matched = reg.captures(text.as_str()).unwrap();
            let (_, [chunk_x, chunk_z, lod]) = matched.extract();
            let chunk_x = i64::from_str(chunk_x).unwrap();
            let chunk_z = i64::from_str(chunk_z).unwrap();
            // the LOD is the depth of the tree. Highest is 10, representing perfect detail
            let lod = u32::from_str(lod).unwrap();
            let data_file = format!("{chunk_x}_{chunk_z}.mtd");
            // get the terrain data
            // check if the file exists
            if Path::exists(data_file.as_ref()) {
                // open the terrain file
                let file = File::open(data_file).expect("failed to open file");
                let mut buf_reader = BufReader::new(file);

                // read the file into a i16 vec
                buffer = Vec::with_capacity(DATUM_COUNT);
                unsafe {
                    buffer.set_len(DATUM_COUNT);
                }
                buf_reader
                    .read_u16_into::<NativeEndian>(&mut buffer[..])
                    .expect("failed to read");
            } else {
                buffer = vec![0; DATUM_COUNT];

                let generator = TerrainGenerator::new();

                // store it so we can reuse it later
                let file = File::create(data_file).expect("failed to open file");
                let mut buf_writer = BufWriter::new(file);

                for i in 0..DATUM_COUNT {
                    let (x, y) = idx_to_coords(i);
                    let x_world_pos = lin_map(0., TILE_DIM as f64, 0., TILE_SIZE, x as f64)
                        + TILE_SIZE * chunk_x as f64;
                    let z_world_pos = lin_map(0., TILE_DIM as f64, 0., TILE_SIZE, y as f64)
                        + TILE_SIZE * chunk_z as f64;
                    buffer[i] =
                        compressed_height(generator.get_height_for(x_world_pos, z_world_pos));
                    // save it for later use
                    buf_writer
                        .write_u16::<NativeEndian>(buffer[i])
                        .expect(format!("failed to write half word {i}").as_str());
                }
            }

            // select the correct data from the file
            // the number of data points is 4^lod
            // the step size is then (total data points) / (data points)
            let compressed_datum_count = 4usize << lod;
            // we only need to send the first compressed_datum_count bytes
            let _ = buffer.split_off(compressed_datum_count);
            let _ = ws_stream.send(Binary(Vec::from(as_u8_slice(&buffer.as_slice()))));
        }
    }

    Ok(())

}

async fn handle_connection(peer: SocketAddr, stream: TcpStream) -> Result<()> {
    let mut ws_stream = accept_async(stream).await.expect("Failed to accept");

    info!("New WebSocket connection: {}", peer);

    while let Some(msg) = ws_stream.next().await {
        let msg = msg?;
        let mut buffer: Vec<u16>;
        if msg.is_text() {
            let text = msg.into_text()?;
            let reg = Regex::new(r"([0-9]+),([0-9]+):([0-9]+)").unwrap();
            let matched = reg.captures(text.as_str()).unwrap();
            let (_, [chunk_x, chunk_z, lod]) = matched.extract();
            let chunk_x = i64::from_str(chunk_x).unwrap();
            let chunk_z = i64::from_str(chunk_z).unwrap();
            // the LOD is the depth of the tree. Highest is 10, representing perfect detail
            let lod = u32::from_str(lod).unwrap();
            let data_file = format!("{chunk_x}_{chunk_z}.mtd");
            // get the terrain data
            // check if the file exists
            if Path::exists(data_file.as_ref()) {
                // open the terrain file
                let file = File::open(data_file).expect("failed to open file");
                let mut buf_reader = BufReader::new(file);

                // read the file into a i16 vec
                buffer = Vec::with_capacity(DATUM_COUNT);
                unsafe {
                    buffer.set_len(DATUM_COUNT);
                }
                buf_reader
                    .read_u16_into::<NativeEndian>(&mut buffer[..])
                    .expect("failed to read");
            } else {
                buffer = vec![0; DATUM_COUNT];

                let generator = TerrainGenerator::new();

                // store it so we can reuse it later
                let file = File::create(data_file).expect("failed to open file");
                let mut buf_writer = BufWriter::new(file);

                for i in 0..DATUM_COUNT {
                    let (x, y) = idx_to_coords(i);
                    let x_world_pos = lin_map(0., TILE_DIM as f64, 0., TILE_SIZE, x as f64)
                        + TILE_SIZE * chunk_x as f64;
                    let z_world_pos = lin_map(0., TILE_DIM as f64, 0., TILE_SIZE, y as f64)
                        + TILE_SIZE * chunk_z as f64;
                    buffer[i] =
                        compressed_height(generator.get_height_for(x_world_pos, z_world_pos));
                    // save it for later use
                    buf_writer
                        .write_u16::<NativeEndian>(buffer[i])
                        .expect(format!("failed to write half word {i}").as_str());
                }
            }

            // select the correct data from the file
            // the number of data points is 4^lod
            // the step size is then (total data points) / (data points)
            let compressed_datum_count = 4usize << lod;
            // we only need to send the first compressed_datum_count bytes
            let _ = buffer.split_off(compressed_datum_count);
            let _ = ws_stream.send(Binary(Vec::from(as_u8_slice(&buffer.as_slice()))));
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let addr = "127.0.0.1:9002";
    let listener = TcpListener::bind(&addr).await.expect("Can't listen");
    println!("Listening on: {}", addr);

    while let Ok((stream, _)) = listener.accept().await {
        let peer = stream
            .peer_addr()
            .expect("connected streams should have a peer address");
        println!("Peer address: {}", peer);

        tokio::spawn(accept_connection(peer, stream));
    }
}
