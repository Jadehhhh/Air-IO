import tartanair as ta

ta.init('/localdisk2/xma29/Air-IO/Tartanair_dataset')

ta.download(
    env='ArchVizTinyHouseDay',
    difficulty=['easy'],
    modality=['imu'],
    camera_name=['lcam_front'],
    unzip=True
)