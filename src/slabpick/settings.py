from typing import List, Optional

from pydantic import BaseModel


class ProcessingSoftware(BaseModel):
    name: str
    version: str


class ProcessingInput(BaseModel):
    in_coords: str
    in_vol: str


class ProcessingOutput(BaseModel):
    out_dir: str
    out_format: List[str]


class ProcessingParametersMinislab(BaseModel):
    extract_shape: List[int]
    voxel_spacing: float
    coords_scale: float
    col_name: Optional[str]
    tomo_type: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    particle_name: Optional[str]
    gallery_shape: List[int]
    one_per_vol: bool
    normalize: bool
    radius: float
    invert: bool


class ProcessingMode(BaseModel):
    live: bool
    t_interval: float
    t_exit: float


class ProcessingConfigMinislab(BaseModel):
    software: ProcessingSoftware
    input: ProcessingInput
    output: ProcessingOutput
    parameters: ProcessingParametersMinislab
    mode: ProcessingMode
