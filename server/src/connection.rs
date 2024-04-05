use bitvec::vec::BitVec;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Copy)]
pub struct TerrainDataPointRequest {
    /// Coordinates on the x-z plane
    pub coordinates: [f32; 2],
    /// Index of the data in the data array, used for quick recall
    pub idx: usize,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct MazeCell {
    pub cell: [i32; 2],
    pub data: BitVec<u32>
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Copy)]
pub struct TerrainDataPoint {
    /// Coordinates on the x-z plane
    pub coordinates: [f32; 2],
    /// Height of the terrain above the x-z plane
    pub height: f32,
    pub gradient: [f32; 2],
    /// Index of the data in the data array, used for quick recall
    pub idx: usize,
}

impl Default for TerrainDataPoint {
    fn default() -> Self {
        return TerrainDataPoint {
            coordinates: [f32::INFINITY, f32::INFINITY],
            height: f32::NAN,
            gradient: [f32::NAN, f32::NAN],
            idx: usize::MAX,
        };
    }
}

// #[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
// pub struct ReqTerrainHeights(pub Vec<TerrainDataPointRequest>);

// #[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
// pub struct TerrainHeights(pub Vec<TerrainDataPoint>);

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum MazeNetworkRequest {
    // TODO: switch to TerrainDataPointRequest
    ReqTerrainHeights(Vec<TerrainDataPoint>),
    ReqMaze(Vec<[i32; 2]>)
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum MazeNetworkResponse {
    TerrainHeights(Vec<TerrainDataPoint>),
    Maze(Vec<MazeCell>)
}

// Use a port of 0 to automatically select a port
pub const CLIENT_PORT: u16 = 0;
pub const SERVER_PORT: u16 = 7665;
