from typing import List, Optional

from pydantic import BaseModel


class ProcessingSoftware(BaseModel):
    name: str
    version: str


# for project_particles command
class ProcessingInputProjectParticles(BaseModel):
    in_coords: str
    in_vol: str


class ProcessingOutputProjectParticles(BaseModel):
    out_dir: str
    out_format: List[str]


class ProcessingParametersProjectParticles(BaseModel):
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
    live: bool
    t_interval: float
    t_exit: float


class ProcessingConfigProjectParticles(BaseModel):
    software: ProcessingSoftware
    input: ProcessingInputProjectParticles
    output: ProcessingOutputProjectParticles
    parameters: ProcessingParametersProjectParticles


# for cs_center_picks command
class ProcessingInputCsCenterPicks(BaseModel):
    cs_file: str
    map_file: str


class ProcessingOutputCsCenterPicks(BaseModel):
    cs_file: str


class ProcessingParametersCsCenterPicks(BaseModel):
    gallery_shape: List[int]


class ProcessingConfigCsCenterPicks(BaseModel):
    software: ProcessingSoftware
    input: ProcessingInputCsCenterPicks
    output: ProcessingOutputCsCenterPicks
    parameters: ProcessingParametersCsCenterPicks


# for cs_map_particles command
class ProcessingInputCsMapParticles(BaseModel):
    copick_json: Optional[str]
    in_star: Optional[str]
    in_star_multiple: Optional[List]
    cs_file: str
    map_file: str


class ProcessingOutputCsMapParticles(BaseModel):
    out_file: str


class ProcessingParametersCsMapParticles(BaseModel):
    col_name: Optional[str]
    particle_name: Optional[str]
    session_id: Optional[str]
    user_id: Optional[str]
    coords_scale: float
    apix: float
    rejected_set: bool


class ProcessingConfigCsMapParticles(BaseModel):
    software: ProcessingSoftware
    input: ProcessingInputCsMapParticles
    output: ProcessingOutputCsMapParticles
    parameters: ProcessingParametersCsMapParticles
