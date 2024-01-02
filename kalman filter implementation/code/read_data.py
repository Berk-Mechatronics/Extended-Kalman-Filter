def read_world(filename):
    landmarks = dict()
    f = open(filename)
    for line in f:
        line_s = line.split('\n')
        line_spl = line_s[0].split(' ')
        landmarks[int(line_spl[0])] = [float(line_spl[1]), float(line_spl[2])]
    return landmarks

def read_sensor_data(filename):
    sensor_readings = dict()
    lm_ids = []
    ranges = []
    bearings = []
    first_time = True
    timestamp = 0
    f = open(filename)
    for line in f:
        line_s = line.split('\n')
        line_spl = line_s[0].split(' ')
        if line_spl[0] == 'ODOMETRY':
            sensor_readings[timestamp, 'odometry'] = {'r1': float(line_spl[1]), 't': float(line_spl[2]), 'r2': float(line_spl[3])}
            if not first_time:
                sensor_readings[timestamp, 'sensor'] = {'id': lm_ids, 'range': ranges, 'bearing': bearings}
                lm_ids = []
                ranges = []
                bearings = []
            timestamp = timestamp + 1
            first_time = False
        if line_spl[0] == 'SENSOR':
            lm_ids.append(int(line_spl[1]))
            ranges.append(float(line_spl[2]))
            bearings.append(float(line_spl[3]))
    sensor_readings[timestamp, 'sensor'] = {'id': lm_ids, 'range': ranges, 'bearing': bearings}
    return sensor_readings
