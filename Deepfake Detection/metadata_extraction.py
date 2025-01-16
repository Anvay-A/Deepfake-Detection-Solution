#Metadata extraction from video files using the Hachoir library

def extract_video_metadata(video_path):
    parser = createParser(video_path)
    if not parser:
        raise ValueError(f"Unable to parse the video: {video_path}")
    metadata = extractMetadata(parser)
    if not metadata:
        raise ValueError(f"No metadata found in the video: {video_path}")
    metadata_dict = {item.key: item.value for item in metadata.exportPlaintext()}
    return metadata_dict
