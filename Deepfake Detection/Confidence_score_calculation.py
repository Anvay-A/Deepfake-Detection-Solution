# This is the file that contains the function that calculates the confidence score of a video based on the scores of 
# the frames, metadata and audio-visual synchronization. This function will be called in main file of the training
# model to calculate the confidence score of a video.

def calculate_confidence(frames, metadata_score, av_sync_score):
    frame_scores = [model.predict(frame) for frame in frames]
    avg_frame_score = sum(frame_scores) / len(frame_scores)
    return (avg_frame_score + metadata_score + av_sync_score) / 3