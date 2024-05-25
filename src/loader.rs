use crate::maze_loader::{MazeResponseReceiver, MazeResponseSender, MazeUpdateProcHandle};
use crate::terrain_loader::{TerrainResponseReceiver, TerrainResponseSender};
use bevy::prelude::{Commands, Deref, ResMut, Resource};
use bevy_tokio_tasks::TokioTasksRuntime;
use crossbeam_channel::{unbounded, Receiver, Sender};
use futures_lite::StreamExt;
use futures_util::{poll, SinkExt};
use postcard::{from_bytes, to_stdvec};
use server::connection::MazeNetworkResponse::TerrainHeights;
use server::connection::{
    CompressedMazeComponent, MazeNetworkRequest, MazeNetworkResponse, TerrainDataPoint,
};
use std::task::Poll;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message::Binary;
use url::Url;

#[derive(Resource, Deref)]
pub struct ServerRequestSender(pub Sender<MazeNetworkRequest>);

#[derive(Resource, Deref)]
pub struct ServerRequestReceiver(pub Receiver<MazeNetworkRequest>);

pub fn setup_websocket(mut commands: Commands, runtime: ResMut<TokioTasksRuntime>) {
    let (request_sender, request_receiver) = unbounded::<MazeNetworkRequest>();
    let (terrain_response_sender, terrain_response_receiver) = unbounded::<Vec<TerrainDataPoint>>();
    let (maze_response_sender, maze_response_receiver) =
        unbounded::<Vec<CompressedMazeComponent>>();

    let req_rec = request_receiver.clone();
    let terrain_resp_sen = terrain_response_sender.clone();
    let maze_resp_sen = maze_response_sender.clone();

    commands.insert_resource::<MazeUpdateProcHandle>(MazeUpdateProcHandle(
        runtime.spawn_background_task(|_ctx| async move {
            let (mut socket, other) =
                connect_async(Url::parse("ws://127.0.0.1:9002").expect("Can't connect to URL"))
                    .await
                    .unwrap();
            loop {
                let mut iter = req_rec.try_iter();
                if let Some(message) = iter.next() {
                    socket
                        .send(Binary(to_stdvec(&message).unwrap()))
                        .await
                        .expect("TODO: panic message");
                }

                match poll!(socket.next()) {
                    Poll::Ready(Some(Ok(Binary(bin)))) => {
                        match from_bytes::<MazeNetworkResponse>(&bin).unwrap() {
                            TerrainHeights(fetched) => {
                                terrain_resp_sen
                                    .send(fetched)
                                    .expect("Could not send terrain update");
                            }
                            MazeNetworkResponse::Maze(fetched) => {
                                maze_resp_sen
                                    .send(fetched)
                                    .expect("Could not send maze update");
                            }
                        }
                    }
                    _ => {
                        // nothing!
                    }
                }
            }
        }),
    ));

    commands.insert_resource::<ServerRequestSender>(ServerRequestSender(request_sender));
    commands.insert_resource::<ServerRequestReceiver>(ServerRequestReceiver(request_receiver));

    commands
        .insert_resource::<TerrainResponseSender>(TerrainResponseSender(terrain_response_sender));
    commands.insert_resource::<TerrainResponseReceiver>(TerrainResponseReceiver(
        terrain_response_receiver,
    ));

    commands.insert_resource::<MazeResponseSender>(MazeResponseSender(maze_response_sender));
    commands.insert_resource::<MazeResponseReceiver>(MazeResponseReceiver(maze_response_receiver));
}
