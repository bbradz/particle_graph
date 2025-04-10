import json2fr

JSON_PATH = 'model/sm.json'
MODEL_PATH = 'model/particles.fr'
sm_data = json2fr.read_json(JSON_PATH)

scalars, fermions, vector_bosons = json2fr.read_particles(sm_data)

fermions = json2fr.read_multiplets(sm_data, scalars, fermions, vector_bosons)

json2fr.write_particles(fermions, MODEL_PATH)