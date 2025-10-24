import json


def convert_python_dict_to_math_dict(python_dict, indent = 4):
    math_dict = []
    for key, value in python_dict.items():
        # if value is not None:
        if key == "Value" and value is None:
            continue
        if isinstance(value, str) and value != "Automatic" and "Dependence" not in key and "Mass" not in key and "OutputName" not in key and "LesHouches" not in key:
            math_dict.append(f"{key} -> \"{value}\"")
        elif isinstance(value, list):
            if isinstance(value[0], str):
                list_str = [f"\"{item}\"" for item in value]
                math_str = ",".join(list_str)
                math_dict.append(f"{key} -> {{{math_str}}}")
            else:
                math_dict.append(f"{key} -> {{{str(value)[1:-1]}}}")
        else:
            math_dict.append(f"{key} -> {value}")
    return '{' + f',\n{" " * indent}'.join(math_dict) + '}'


def write_particles(all_particles, output_file):
    with open(output_file, "w") as f:
        f.write("(*-------------------------------------------*)\n")
        f.write("(*                Particles                 *)\n")
        f.write("(*-------------------------------------------*)\n")
        f.write("\n")
    for section in all_particles:
        particles = []
        for particle in all_particles[section]:
            particle_name = f"{particle},"
            particle_dict = all_particles[section][particle].copy()
            
            for key in ["chirality", "class_members", "location", "left", "right"]:
                particle_dict.pop(key, None)

            particle_info = []
            quote_keys = ["Description", "LaTeX", "OutputName"]
            for key, value in particle_dict.items():
                if not isinstance(value, list):
                    if key in quote_keys:
                        particle_info.append(f"{key} -> \"{value}\"")
                    else:
                        particle_info.append(f"{key} -> {value}")
                else:
                    list_str = []
                    for item in value:
                        if key in quote_keys:
                            item = f"\"{item}\""
                        else:
                            item = str(item)
                        list_str.append(item)
                    math_str = ", ".join(list_str)
                    particle_info.append(f"{key} -> {{{math_str}}}")

            particle_info = '{' + f',\n{" " * 11}'.join(particle_info) + '}'            
            particle = f"    {{{particle_name:4} {particle_info}}}"
            particles.append(particle)

        with open(output_file, "a") as f:
            f.write(f"{section} = {{\n")
            f.write(",\n".join(particles))
            f.write("\n")
            f.write("};\n\n")


def write_parameters(parameters, output_file, indent = 11):
    with open(output_file, "a") as f:
        f.write("(*-------------------------------------------*)\n")
        f.write("(*                Parameters                 *)\n")
        f.write("(*-------------------------------------------*)\n")
        f.write("\n")
    all_parameters = []
    for param in parameters:
        parameter_name = f"{param},"
        parameter_info = []
        quote_keys = ["Description", "LaTeX", ]
        for key, value in parameters[param].items():
            if key == "Value" and value is None:
                continue
            if key in quote_keys:
                parameter_info.append(f"{key} -> \"{value}\"")
            else:
                parameter_info.append(f"{key} -> {value}")

        parameter_info = '{' + f',\n{" " * indent}'.join(parameter_info) + '}'
        parameter = f"    {{{parameter_name:4} {parameter_info}}}"
        all_parameters.append(parameter)
    with open(output_file, "a") as f:
        f.write(f"ParameterDefinitions = {{\n")
        f.write(",\n\n".join(all_parameters))
        f.write("\n")
        f.write("};\n\n")


def write_SPheno(match_conditions, 
                 parameters_to_solve_tadpoles,
                 MINPAR,
                 output_file,
                 list_decay_particles,
                 low_energy = True,
                 non_standard_yukawas = False,
                 tree_level_unitarity_limits = True
                 ):
    MINPAR = list(MINPAR.values())
    with open(output_file, "w") as f:
        f.write("(*-------------------------------------------*)\n")
        f.write("(*                   SPheno                  *)\n")
        f.write("(*-------------------------------------------*)\n")
        f.write("\n")
        if low_energy:
            f.write("OnlyLowEnergySPheno = True;\n\n")
        if tree_level_unitarity_limits:
            f.write("AddTreeLevelUnitarityLimits=True;\n\n")
        if non_standard_yukawas:
            f.write("DEFINITION[UseNonStandardYukwas] = True;\n")
            f.write("DEFINITION[NonStandardYukawasRelations] = {};\n\n")
        
        # MINPAR
        minimal_parameters = [f"{{{idx+1}, {param.OutputName}INPUT}}" for idx, param in enumerate(MINPAR)]
        f.write(f"MINPAR={{\n")
        f.write(",\n".join(minimal_parameters))
        f.write("\n")
        f.write("};\n\n")
        
        # Parameters to solve tadpoles
        f.write(f"ParametersToSolveTadpoles={{{', '.join(parameters_to_solve_tadpoles)}}};\n\n")

        # Boundary Low Scale Input
        low_scale_input = [f"{{{param.name}, {param.OutputName}INPUT}}" for idx, param in enumerate(MINPAR)]
        f.write("BoundaryLowScaleInput={\n")
        f.write(",\n".join(low_scale_input))
        f.write("\n")
        f.write("};\n\n")

        # Matching conditions
        f.write("DEFINITION[MatchingConditions]= {\n")
        f.write("{" + "},\n{".join(match_conditions) + "}\n")
        f.write("};\n\n")

        # List Decay Particles
        f.write("ListDecayParticles = {")
        f.write(",".join(list_decay_particles))
        f.write("};\n\n")

        # DefaultInputValues
        f.write("DefaultInputValues={\n")
        minimal_parameters = [f"{param.OutputName}INPUT -> {param.Value}" for param in MINPAR]
        f.write(",\n".join(minimal_parameters))
        f.write("\n};\n\n")

        # Renormalization conditions
        f.write("RenConditionsDecays={\n")
        f.write("{dCosTW, 1/2*Cos[ThetaW] * (PiVWp/(MVWp^2) - PiVZ/(mVZ^2)) },\n")
        f.write("{dSinTW, -dCosTW/Tan[ThetaW]},\n")
        f.write("{dg2, 1/2*g2*(derPiVPheavy0 + PiVPlightMZ/MVZ^2 - (-(PiVWp/MVWp^2) + PiVZ/MVZ^2)/Tan[ThetaW]^2 + (2*PiVZVP*Tan[ThetaW])/MVZ^2)  },\n")
        f.write("{dg1, dg2*Tan[ThetaW]+g2*dSinTW/Cos[ThetaW]- dCosTW*g2*Tan[ThetaW]/Cos[ThetaW]}\n")
        f.write("};\n\n")
        


if __name__ == "__main__":
    with open("token_to_model/sm_parameters.json", "r") as f:
        all_parameters = json.load(f)
    write_parameters(all_parameters, "test.txt")