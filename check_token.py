from rl.grammar import GrammarMasker

grammar = GrammarMasker()
state = grammar.initial_state()

token = "ITRACT_ID_1 TYPE_YUKAWA FIELD_ID_1 TYPE_FIELD_fermion DIM_1 GEN_1 SELF_CONJ_TRUE CHIRALITY_left SU3C_REP_1 SU2L_REP_1 U1Y_CHARGE_-1 QN_L_0 QN_B_0 PARTICLE_ID_1 TYPE_PARTICLE_fermion MASS_1e-9 CHARGE_-1 END_PARTICLE END_FIELD END_ITRACT EOS"

print(grammar.step(state, token))