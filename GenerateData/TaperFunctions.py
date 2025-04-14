from genesis.version4 import Genesis4, Write
import genesis.version4 as g4
import numpy as np

def parse_genesis4_lattice_file(filename, keys = ['Undulator', 'Quadrupole', 'Corrector', 'Phaseshifter', ' Drift', 'Chicane', 'Marker', 'Line']):
    """
    Parse genesis4 lattice file to an dictionary
    """


    with open(filename) as f:
        lines = f.readlines()
    output = {}
    for key in keys:
        output[key] = []

    ind = 0
    while ind  < len(lines):
        line = lines[ind]
        for key in keys:
            if key.upper() in line.upper() and '#' not in line:
                ele_str = line + ' '
                while '}' not in line:
                    ind += 1
                    line = lines[ind]
                    ele_str += line + ' '

                output[key].append(ele_str)
        ind += 1
    return output    


def gettaper(n_und, Kstart, dKbyK_linear, dKbyK_quadratic, ustart, u_linear_stop, u_quadratic_stop):
    """
    Compute start and end K for each undulator section with separate linear and quadratic tapers.

    Parameters:
        n_und (int): Total number of undulator sections.
        Kstart (float): Initial undulator parameter K.
        dKbyK_linear (float): Fractional tapering for the linear taper.
        dKbyK_quadratic (float): Fractional tapering for the quadratic taper.
        ustart (int): Start index for tapering.
        u_linear_stop (int): Stop index for linear taper (last section before quadratic).
        u_quadratic_stop (int): Stop index for quadratic taper.
    
    Returns:
        list: A list of [K_start, K_end] pairs for each undulator section.
    """
    Klist = []

    # Fill initial undulators with constant Kstart (before tapering starts)
    for _ in range(ustart):
        Klist.append([Kstart, Kstart])

    # Compute final K values for linear and quadratic tapers
    K_linear_end = Kstart * (1 - dKbyK_linear)
    K_quadratic_start = K_linear_end  # Quadratic taper should begin where linear ended
    K_quadratic_end = K_quadratic_start * (1 - dKbyK_quadratic)

    # Determine the number of undulators for each taper segment
    nund_linear = max(0, u_linear_stop - ustart)  # Linear taper stops at `u_linear_stop`
    nund_quadratic = max(0, u_quadratic_stop - u_linear_stop - 1)  # Quadratic starts after `u_linear_stop`

    # Linear taper section
    if nund_linear > 0:
        K_linear = np.linspace(Kstart, K_linear_end, nund_linear + 1)
        for i in range(nund_linear):
            Klist.append([K_linear[i], K_linear[i + 1]])

    # Quadratic taper section (starts **after** `u_linear_stop`)
    if nund_quadratic > 0:
        alpha = (K_quadratic_end - K_quadratic_start) / nund_quadratic**2
        K_quadratic = K_quadratic_start + alpha * np.arange(1, nund_quadratic + 2)**2  # Start from index 1
        for i in range(nund_quadratic):
            Klist.append([K_quadratic[i], K_quadratic[i + 1]])

    # Fill remaining undulators with constant Kstart (after `u_quadratic_stop`)
    for _ in range(u_quadratic_stop, n_und):
        Klist.append([Kstart, Kstart])

    return Klist


def write_constant_sec(usegname, K, nwig, uperiod):
    line = usegname + ': Undulator = {lambdau=' + str(uperiod) + ', nwig=' + str(nwig) + ', aw=' + str(K) + '};\n'
    return [line]


def write_linear_taper_sec(usegname, Kstart, Kend, nwig, uperiod):
    """
    Within an undulator, linearly change the taper of each period
    """
    Klist = np.linspace(Kstart, Kend, nwig)
    und_line = []
    ele_names = []
    for i in range(nwig):
        ele_name = usegname + str(i)
        line = usegname + str(i) + ': Undulator = {lambdau=' + str(uperiod) + ', nwig=1, aw=' + str(Klist[i]) + '};\n'
        und_line.append(line)
        ele_names.append(ele_name)
    return und_line, ele_names

def write_undulator(usegname_list, Kstart, dKbyK_linear, dKbyK_quadratic, ustart, u_linear_stop, u_quadratic_stop, nwig, uperiod):
    """
    Modify undulator sections with separate linear and quadratic tapers.
    
    Parameters:
        usegname_list (list): List of undulator section names.
        Kstart (float): Initial K value.
        dKbyK_linear (float): Linear taper strength.
        dKbyK_quadratic (float): Quadratic taper strength.
        ustart (int): Start index for tapering.
        u_linear_stop (int): End of linear taper, start of quadratic taper.
        u_quadratic_stop (int): End of quadratic taper.
        nwig (int): Number of undulator periods per section.
        uperiod (float): Undulator period length.
    
    Returns:
        list: Modified undulator lattice lines.
    """
    Klist = gettaper(
        n_und=len(usegname_list),
        Kstart=Kstart,
        dKbyK_linear=dKbyK_linear,
        dKbyK_quadratic=dKbyK_quadratic,
        ustart=ustart,
        u_linear_stop=u_linear_stop,
        u_quadratic_stop=u_quadratic_stop
    )

    lines = []

    for count, Kpar in enumerate(Klist):
        usegname = usegname_list[count]
        
        if len(Kpar) == 1:
            lines.extend(write_constant_sec(usegname, Kpar[0], nwig, uperiod))
        else:
            und_line, ele_names = write_linear_taper_sec(usegname + '_s', Kpar[0], Kpar[1], nwig, uperiod)
            lines.extend(und_line)
            line = usegname + ': Line = {' + ele_names[0]
            for ele_name in ele_names[1:]:
                line += ', ' + ele_name
            line += '};\n'
            lines.append(line)

    return lines

def apply_taper(self, Kstart, dKbyK_linear, dKbyK_quadratic, ustart, u_linear_stop, u_quadratic_stop, nwig, uperiod):
    """
    Read from an existing lattice file and modify the undulator with separate linear and quadratic tapers.
    
    Parameters:
        Kstart (float): Initial undulator parameter K.
        dKbyK_linear (float): Fractional taper for the linear region.
        dKbyK_quadratic (float): Fractional taper for the quadratic region.
        ustart (int): Start index for tapering.
        u_linear_stop (int): End of linear taper, start of quadratic taper.
        u_quadratic_stop (int): End of quadratic taper.
        nwig (int): Number of undulator periods per section.
        uperiod (float): Undulator period length.
    """
    
    output_lines = []

    # Parse the lattice file
    lattice = parse_genesis4_lattice_file(self.input.lattice.filename)

    # Retain all non-undulator elements
    for key in lattice:
        if key not in ['Undulator', 'Line']:
            output_lines.extend(lattice[key])

    # Extract undulator section names
    usegname_list = [und.split(':')[0] for und in lattice['Undulator']]

    # Apply tapering with both linear and quadratic sections
    lines = write_undulator(
        usegname_list, 
        Kstart, 
        dKbyK_linear, 
        dKbyK_quadratic, 
        ustart, 
        u_linear_stop, 
        u_quadratic_stop, 
        nwig, 
        uperiod
    )

    # Append modified undulator sections and original lattice lines
    output_lines.extend(lines)
    output_lines.extend(lattice['Line'])

    # Update the Genesis input lattice
    self.input.lattice = g4.Lattice.from_contents(''.join(output_lines))

