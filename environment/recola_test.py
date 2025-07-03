import sys
sys.path.append('environment/')

import pyrecola

# The standard output is selected
pyrecola.set_output_file_rcl('*')

# Let's print the squared amplitude
pyrecola.set_print_level_squared_amplitude_rcl(1)

pyrecola.define_process_rcl(1, 'u u~ -> g g tau+ tau-', 'NLO')

pyrecola.generate_processes_rcl()

p1 = [4000.0000000000,    0.0000000000,    0.0000000000, 4000.0000000000]
p2 = [4000.0000000000,    0.0000000000,    0.0000000000,-4000.0000000000]
p3 = [2387.4445571379,-2131.7219821216,  677.6712380335, -834.5145879427]
p4 = [2084.0108209587, 1206.0274745508, 1266.0449626178,-1133.8999008430]
p5 = [1954.1326742459, -173.3442838631, -836.2617619034, 1757.6269608155]
p6 = [1574.4119476575, 1099.0387914340,-1107.4544387478,  210.7875279701]
p = [p1, p2, p3, p4, p5, p6]

pyrecola.compute_process_rcl(1,p,'NLO')

pyrecola.reset_recola_rcl()