"""Public payload definitions shared by streaming and prediction services."""

STREAM_CORE_COLUMNS = {
    "building_id", "timestamp", "meter_type", "meter_reading", "site_id", "timezone", "x", "y",
}

# These are all available building attributes from lead_buildings.csv that are
# safe to expose. site_id and timezone deliberately remain internal.
PUBLIC_BUILDING_METADATA_COLUMNS = (
    "sub_primaryspaceusage",
    "primaryspaceusage",
    "sqm",
    "sqft",
    "yearbuilt",
    "occupants",
    "real_lat",
    "real_lng",
    "local_x_m",
    "local_y_m",
)