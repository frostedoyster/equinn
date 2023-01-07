def get_spherical_expansion_ghost(all_species, l_max, q_max):

    ghost = {}
    for a_i in all_species:
        for l in range(l_max+1):
            ghost[(a_i, l, 1)] = q_max[l]

    return ghost
