use futures_util::{SinkExt, StreamExt};
use log::*;
use postcard::{from_bytes, to_stdvec};
use server::connection::TerrainDataPoint;
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
            println!("Intercepted message!");
            let mut to_get = from_bytes::<Vec<TerrainDataPoint>>(&bin).unwrap();
            println!("Fetching {}", to_get.len());
            for point in &mut to_get {
                // println!("BEFORE {:?} {:?} {} {}", point.coordinates, point.gradient, point.height, point.idx);
                let [x, y] = point.coordinates;
                point.height = gen.get_height_for(x as f64, y as f64) as f32;
                point.gradient = gen.get_gradient(x as f64, y as f64).as_vec2().to_array();
                // println!("AFTER {:?} {:?} {} {}", point.coordinates, point.gradient, point.height, point.idx);
            }
            ws_stream.send(Binary(to_stdvec(&to_get).unwrap())).await?;
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    env_logger::init();

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
