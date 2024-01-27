use bevy::prelude::*;
use lightyear::prelude::server::*;
use lightyear::prelude::{Io, IoConfig, LinkConditionerConfig, PingConfig};
use log::*;
use std::collections::HashMap;
use std::io::Read;
use std::net::{Ipv4Addr, SocketAddr};
use std::str::FromStr;
use std::time::Duration;

use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::DefaultPlugins;
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use clap::{Parser, ValueEnum};

use lightyear::netcode::ClientId;
use lightyear::prelude::TransportConfig;
use server::connection::*;
use server::terrain_gen::TerrainGenerator;

pub struct MyServerPlugin {
    pub(crate) headless: bool,
    pub(crate) port: u16,
    pub(crate) transport: Transports,
}

impl Plugin for MyServerPlugin {
    fn build(&self, app: &mut App) {
        let server_addr = SocketAddr::new(Ipv4Addr::UNSPECIFIED.into(), self.port);
        let netcode_config = NetcodeConfig::default()
            .with_protocol_id(PROTOCOL_ID)
            .with_key(KEY);
        let link_conditioner = LinkConditionerConfig {
            incoming_latency: Duration::from_millis(200),
            incoming_jitter: Duration::from_millis(20),
            incoming_loss: 0.05,
        };
        let transport = match self.transport {
            Transports::Udp => TransportConfig::UdpSocket(server_addr),
            Transports::Webtransport => TransportConfig::WebTransportServer {
                server_addr,
                certificate: Certificate::self_signed(&["localhost"]),
            },
        };
        let io =
            Io::from_config(IoConfig::from_transport(transport).with_conditioner(link_conditioner));
        let config = ServerConfig {
            shared: shared_config().clone(),
            netcode: netcode_config,
            ping: PingConfig::default(),
        };
        let plugin_config = PluginConfig::new(config, io, protocol());
        app.add_plugins(ServerPlugin::new(plugin_config));
        // app.add_plugins(shared::SharedPlugin);
        app.init_resource::<Global>();
        app.add_systems(Startup, init);
        // the physics/FixedUpdates systems that consume inputs should be run in this set
        // app.add_systems(FixedUpdate, movement.in_set(FixedUpdateSet::Main));
        app.add_systems(Update, get_terrain_heights);
        app.add_systems(Update, handle_connections);
    }
}

fn get_terrain_heights(
    mut server: ResMut<Server>,
    mut terrain_heights_reader: EventReader<MessageEvent<ReqTerrainHeights>>,
) {
    let gen = TerrainGenerator::new();
    for message in terrain_heights_reader.read() {
        println!("Intercepted message!");
        let client_id = message.context();
        let mut terrain_heights = Vec::<TerrainDataPoint>::new();
        for data in &message.message().0 {
            let [x, y] = data.coordinates;
            terrain_heights.push(TerrainDataPoint {
                coordinates: [x, y],
                height: gen.get_height_for(x as f64, y as f64) as f32,
                idx: data.idx,
                gradient: gen.get_gradient(x as f64, y as f64).as_vec2().to_array()
            });
        }
        server
            .send_message::<Channel1, TerrainHeights>(*client_id, TerrainHeights(terrain_heights))
            .expect("TODO: panic message");
    }
}

#[derive(Resource, Default)]
pub(crate) struct Global {
    pub client_id_to_entity_id: HashMap<ClientId, Entity>,
}

pub(crate) fn init(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
    commands.spawn(TextBundle::from_section(
        "Server",
        TextStyle {
            font_size: 30.0,
            color: Color::WHITE,
            ..default()
        },
    ));
}

/// Server connection system, create a player upon connection
pub(crate) fn handle_connections(
    mut connections: EventReader<ConnectEvent>,
    mut disconnections: EventReader<DisconnectEvent>,
    mut global: ResMut<Global>,
    mut commands: Commands,
) {
    for connection in connections.read() {
        let client_id = connection.context();
        // Generate pseudo random color from client id.
        let h = (((client_id * 30) % 360) as f32) / 360.0;
        let s = 0.8;
        let l = 0.5;
        let entity = commands.spawn(PlayerBundle::new(
            *client_id,
            Vec2::ZERO,
            Color::hsl(h, s, l),
        ));
        // Add a mapping from client id to entity id
        global
            .client_id_to_entity_id
            .insert(*client_id, entity.id());
    }
    for disconnection in disconnections.read() {
        let client_id = disconnection.context();
        if let Some(entity) = global.client_id_to_entity_id.remove(client_id) {
            commands.entity(entity).despawn();
        }
    }
}

// async fn get_from_coordinates(peer: SocketAddr, stream: TcpStream) -> Result<()> {
//     let mut ws_stream = accept_async(stream).await.expect("Failed to accept");
//
//     info!("New WebSocket connection: {}", peer);
//
//     while let Some(msg) = ws_stream.next().await {
//         let msg = msg?;
//         if msg.is_binary() {
//             let bin = msg.into_data()?;
//         }
//
//         let mut buffer: Vec<u16>;
//         if msg.is_text() {
//             let text = msg.into_text()?;
//             let reg = Regex::new(r"([0-9]+),([0-9]+):([0-9]+)").unwrap();
//             let matched = reg.captures(text.as_str()).unwrap();
//             let (_, [chunk_x, chunk_z, lod]) = matched.extract();
//             let chunk_x = i64::from_str(chunk_x).unwrap();
//             let chunk_z = i64::from_str(chunk_z).unwrap();
//             // the LOD is the depth of the tree. Highest is 10, representing perfect detail
//             let lod = u32::from_str(lod).unwrap();
//             let data_file = format!("{chunk_x}_{chunk_z}.mtd");
//             // get the terrain data
//             // check if the file exists
//             if Path::exists(data_file.as_ref()) {
//                 // open the terrain file
//                 let file = File::open(data_file).expect("failed to open file");
//                 let mut buf_reader = BufReader::new(file);
//
//                 // read the file into a i16 vec
//                 buffer = Vec::with_capacity(DATUM_COUNT);
//                 unsafe {
//                     buffer.set_len(DATUM_COUNT);
//                 }
//                 buf_reader
//                     .read_u16_into::<NativeEndian>(&mut buffer[..])
//                     .expect("failed to read");
//             } else {
//                 buffer = vec![0; DATUM_COUNT];
//
//                 let generator = TerrainGenerator::new();
//
//                 // store it so we can reuse it later
//                 let file = File::create(data_file).expect("failed to open file");
//                 let mut buf_writer = BufWriter::new(file);
//
//                 for i in 0..DATUM_COUNT {
//                     let (x, y) = idx_to_coords(i);
//                     let x_world_pos = lin_map(0., TILE_DIM as f64, 0., TILE_SIZE, x as f64)
//                         + TILE_SIZE * chunk_x as f64;
//                     let z_world_pos = lin_map(0., TILE_DIM as f64, 0., TILE_SIZE, y as f64)
//                         + TILE_SIZE * chunk_z as f64;
//                     buffer[i] =
//                         compressed_height(generator.get_height_for(x_world_pos, z_world_pos));
//                     // save it for later use
//                     buf_writer
//                         .write_u16::<NativeEndian>(buffer[i])
//                         .expect(format!("failed to write half word {i}").as_str());
//                 }
//             }
//
//             // select the correct data from the file
//             // the number of data points is 4^lod
//             // the step size is then (total data points) / (data points)
//             let compressed_datum_count = 4usize << lod;
//             // we only need to send the first compressed_datum_count bytes
//             let _ = buffer.split_off(compressed_datum_count);
//             let _ = ws_stream.send(Binary(Vec::from(as_u8_slice(&buffer.as_slice()))));
//         }
//     }
//
//     Ok(())
// }
//
// async fn handle_connection(peer: SocketAddr, stream: TcpStream) -> Result<()> {
//     let mut ws_stream = accept_async(stream).await.expect("Failed to accept");
//
//     info!("New WebSocket connection: {}", peer);
//
//     while let Some(msg) = ws_stream.next().await {
//         let msg = msg?;
//         let mut buffer: Vec<u16>;
//         if msg.is_text() {
//             let text = msg.into_text()?;
//             let reg = Regex::new(r"([0-9]+),([0-9]+):([0-9]+)").unwrap();
//             let matched = reg.captures(text.as_str()).unwrap();
//             let (_, [chunk_x, chunk_z, lod]) = matched.extract();
//             let chunk_x = i64::from_str(chunk_x).unwrap();
//             let chunk_z = i64::from_str(chunk_z).unwrap();
//             // the LOD is the depth of the tree. Highest is 10, representing perfect detail
//             let lod = u32::from_str(lod).unwrap();
//             let data_file = format!("{chunk_x}_{chunk_z}.mtd");
//             // get the terrain data
//             // check if the file exists
//             if Path::exists(data_file.as_ref()) {
//                 // open the terrain file
//                 let file = File::open(data_file).expect("failed to open file");
//                 let mut buf_reader = BufReader::new(file);
//
//                 // read the file into a i16 vec
//                 buffer = Vec::with_capacity(DATUM_COUNT);
//                 unsafe {
//                     buffer.set_len(DATUM_COUNT);
//                 }
//                 buf_reader
//                     .read_u16_into::<NativeEndian>(&mut buffer[..])
//                     .expect("failed to read");
//             } else {
//                 buffer = vec![0; DATUM_COUNT];
//
//                 let generator = TerrainGenerator::new();
//
//                 // store it so we can reuse it later
//                 let file = File::create(data_file).expect("failed to open file");
//                 let mut buf_writer = BufWriter::new(file);
//
//                 for i in 0..DATUM_COUNT {
//                     let (x, y) = idx_to_coords(i);
//                     let x_world_pos = lin_map(0., TILE_DIM as f64, 0., TILE_SIZE, x as f64)
//                         + TILE_SIZE * chunk_x as f64;
//                     let z_world_pos = lin_map(0., TILE_DIM as f64, 0., TILE_SIZE, y as f64)
//                         + TILE_SIZE * chunk_z as f64;
//                     buffer[i] =
//                         compressed_height(generator.get_height_for(x_world_pos, z_world_pos));
//                     // save it for later use
//                     buf_writer
//                         .write_u16::<NativeEndian>(buffer[i])
//                         .expect(format!("failed to write half word {i}").as_str());
//                 }
//             }
//
//             // select the correct data from the file
//             // the number of data points is 4^lod
//             // the step size is then (total data points) / (data points)
//             let compressed_datum_count = 4usize << lod;
//             // we only need to send the first compressed_datum_count bytes
//             let _ = buffer.split_off(compressed_datum_count);
//             let _ = ws_stream.send(Binary(Vec::from(as_u8_slice(&buffer.as_slice()))));
//         }
//     }
//
//     Ok(())
// }

// Run with
// - `cargo run --example simple_box -- server`
// - `cargo run --example simple_box -- client -c 1`

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let mut app = App::new();
    setup(&mut app, cli);

    app.run();
}

#[derive(Parser, PartialEq, Debug)]
enum Cli {
    Server {
        #[arg(long, default_value = "false")]
        headless: bool,

        #[arg(short, long, default_value = "false")]
        inspector: bool,

        #[arg(short, long, default_value_t = SERVER_PORT)]
        port: u16,

        #[arg(short, long, value_enum, default_value_t = Transports::Udp)]
        transport: Transports,
    },
}

fn setup(app: &mut App, cli: Cli) {
    match cli {
        Cli::Server {
            headless,
            inspector,
            port,
            transport,
        } => {
            let server_plugin = MyServerPlugin {
                headless,
                port,
                transport,
            };
            if !headless {
                app.add_plugins(DefaultPlugins.build().disable::<LogPlugin>());
            } else {
                app.add_plugins(MinimalPlugins);
            }
            if inspector {
                app.add_plugins(WorldInspectorPlugin::new());
            }
            app.add_plugins(server_plugin);
        }
    }
}
