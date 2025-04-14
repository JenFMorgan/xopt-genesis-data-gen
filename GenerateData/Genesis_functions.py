from genesis.version4 import Genesis4, Write
import genesis.version4 as g4
from genesis import tools
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import c, e, epsilon_0
import scipy
from scipy.optimize import curve_fit

import json
import numpy as np
from pmd_beamphysics import ParticleGroup
from pathlib import Path
import os
from pathlib import Path
import os
from pmd_beamphysics import ParticleGroup
h = scipy.constants.value("Planck constant in eV/Hz")
import hashlib
from TaperFunctions import *
import h5py
Genesis4.apply_taper = apply_taper
import time
import h5py
import hashlib
import shutil
from genesis.version4 import ProfileFile
class NpEncoder(json.JSONEncoder):
    """
    Custom encoder to serialize non-standard data types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, ParticleGroup):
            # Convert ParticleGroup into a JSON-serializable dictionary
            return {
                key: getattr(obj, key).tolist() if isinstance(getattr(obj, key), np.ndarray) else getattr(obj, key)
                for key in obj._settable_array_keys + obj._settable_scalar_keys
            }
        elif isinstance(obj, Path):  # Convert PosixPath to string
            return str(obj)
        else:
            return super(NpEncoder, self).default(obj)


import hashlib


import time

def safe_h5_write(filename, data_dict, retries=8, delay=0.2):
    """
    Safely writes data to an HDF5 file with retries to avoid file locking issues.

    Parameters:
    - filename (str): Full path to the HDF5 file to write.
    - data_dict (dict): Dictionary of dataset_name: array to store.
    - retries (int): Number of retry attempts on failure.
    - delay (float): Initial delay between retries (will be increased exponentially).
    """
    for attempt in range(retries):
        try:
            with h5py.File(filename, "w") as f:
                for key, value in data_dict.items():
                    f[key] = value
            return  # success
        except (BlockingIOError, OSError) as e:
            print(f"⚠️ File write error for {filename}: {e} (attempt {attempt+1}/{retries})")
            time.sleep(delay * (attempt + 1))  # exponential backoff
    raise RuntimeError(f"❌ Failed to write {filename} after {retries} retries.")


def generate_beam_fingerprint(input_dict):
    """
    Generates a unique fingerprint based on all relevant beam parameters.

    Parameters:
    - input_dict (dict): Dictionary containing beam and taper parameters.

    Returns:
    - str: A unique fingerprint string.
    """
    keys_to_include = [
        'slen', 'beam_length', 'current', 'gamma0', 'chirp',
        'alphax', 'alphay', 'betax', 'betay',
        'emitx', 'emity',
        'taper', 'taper_quadratic',
        'ustart', 'u_linear_stop', 'u_quadratic_stop'
    ]
    
    # Collect key=value string representations in a consistent order
    parts = []
    for key in keys_to_include:
        val = input_dict.get(key, 'NA')
        parts.append(f"{key}={val}")
    
    params_str = '_'.join(parts)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


def generate_beam_profiles(slen, curr0, gamma0, chirp, beam_length, target_dir, fingerprint, num_points=20000):
    xdata = np.linspace(0, slen, num_points)
    ydata_gamma = gamma0 + (xdata - xdata.mean()) * chirp * 1e6 / 0.511

#    gamma_filename = os.path.join(target_dir, "beam_gamma.h5")
    gamma_filename = os.path.join(target_dir, f"beam_gamma_{fingerprint}.h5")

    
    safe_h5_write(gamma_filename, {"s": xdata, "gamma": ydata_gamma})

    ydata_current = np.zeros_like(xdata)
    
    if beam_length is None or beam_length >= slen:
        ydata_current[:] = curr0
    else:
        center = slen / 2
        half_beam = beam_length / 2
        mask = (xdata >= (center - half_beam)) & (xdata <= (center + half_beam))
        ydata_current[mask] = curr0

    current_filename = os.path.join(target_dir,  f"beam_current_{fingerprint}.h5")
    safe_h5_write(current_filename, {"s": xdata, "current": ydata_current})

    return gamma_filename, current_filename

def generate_beam_profiles_hold(slen, curr0, gamma0, chirp, beam_length, fingerprint, num_points=20000):
    """
    Generates beam current and gamma profiles, saving them to HDF5 files with a unique fingerprint.

    Parameters:
    - slen (float): Total beam length.
    - curr0 (float): Peak current value.
    - gamma0 (float): Initial gamma value.
    - h (float): Chirp parameter.
    - fingerprint (str): Unique identifier for filenames.
    - num_points (int, optional): Number of points in the profile (default is 20000).

    Outputs:
    - Saves HDF5 files: "<fingerprint>_beam_current.h5" and "<fingerprint>_beam_gamma.h5"
    """

    # Generate x-axis values
    xdata = np.linspace(0, slen, num_points)

    # Gamma profile (linear chirp)
    ydata_gamma = gamma0 + (xdata - xdata.mean()) * chirp * 1e6 / 0.511  # Convert chirp to proper scaling

    # Save gamma profile
    gamma_filename = f"{fingerprint}_beam_gamma.h5"
    with h5py.File(gamma_filename, "w") as file:
        file["s"] = xdata
        file["gamma"] = ydata_gamma
    print(f"Saved: {gamma_filename}")

    ydata_current = np.zeros_like(xdata)

    if beam_length is None or beam_length >= slen:
        ydata_current[:] = curr0
    else:
        center = slen / 2
        half_beam = beam_length / 2
        mask = (xdata >= (center - half_beam)) & (xdata <= (center + half_beam))
        ydata_current[mask] = curr0

    # Save current profile
    current_filename = f"{fingerprint}_beam_current.h5"
    with h5py.File(current_filename, "w") as file:
        file["s"] = xdata
        file["current"] = ydata_current
    print(f"Saved: {current_filename}")

    return gamma_filename, current_filename  # Return filenames for reference


def Getfingerprint(keyed_data, digest_size=16):
    """
    Creates a cryptographic fingerprint from keyed data, including ParticleGroup objects.
    
    Uses JSON dumps to form strings and the blake2b algorithm to hash.

    Parameters
    ----------
    keyed_data : dict
        Dictionary with the keys to generate a fingerprint. Can include ParticleGroup.
    digest_size : int, optional
        Digest size for blake2b hash code, by default 16

    Returns
    -------
    str
        The hexadecimal digest
    """
    h = hashlib.blake2b(digest_size=digest_size)
    s = json.dumps(keyed_data, sort_keys=True, cls=NpEncoder).encode()
    h.update(s)
    return h.hexdigest()



import os
import h5py
import numpy as np

h = scipy.constants.value("Planck constant in eV/Hz")

def get_spectrum(G):
    """
    Calculate the spectrum from the Genesis output object G and find the
    peak spectral fluence, the photon energy at the peak, and the FWHM.

    Parameters:
    G (Genesis object): The Genesis object containing simulation output data.

    Returns:
    peak_fluence (float): The maximum spectral fluence.
    peak_energy (float): The photon energy at the peak spectral fluence.
    fwhm_bandwidth (float): The Full Width at Half Maximum (FWHM) of the spectrum in eV.
    """

    print('getting output')
    
    output = G.output
    
    print('got output')
    
    # Extract intensity, phase, and frequency information
    sig = output.field.intensity_farfield[-1, :]
    phi = output.field.phase_farfield[-1, :]
    freq = output.globals.frequency

    # Extract energy information
    energy = output.beam.energy[0, :] * 0.511 * 1e-3
    energy0 = energy[-1]

    # Calculate the signal and spectrum
    signal = np.sqrt(sig) * np.exp(1j * phi)
    spec = np.abs(np.fft.fftshift(np.fft.fft(signal))) ** 2
    norm = energy0 / np.sum(spec) / (freq[1] - freq[0])
    spec = norm * spec

 
    # Convert frequency to photon energy (eV) and compute spectral fluence (J/m^2/eV)
    photon_energies = h * freq  # eV
    spectral_fluence = spec  # Spectral fluence directly from normalized spectrum

    # Analyze spectrum to find peak, peak energy, and FWHM
    #peak_fluence, peak_energy, fwhm_bandwidth = analyze_spectrum(freq, spectral_fluence)
    #plt.plot(freq, spec)
    #plt.plot(freq, gaussian(freq, peak_fluence, peak_energy, fwhm_bandwidth/(2 * np.sqrt(2 * np.log(2)))))
    #print(fwhm_bandwidth, 'FWHM')
    return photon_energies, spectral_fluence


def archive_with_fingerprint(G):
    # Generate fingerprint
    fingerprint = Getfingerprint(G.input.model_dump())

    # Get the current working directory
    current_dir = os.getcwd()

    # Define separate directories for input and output
    input_dir = os.path.join(current_dir, "processed_inputs")
    output_dir = os.path.join(current_dir, "processed_outputs")

    # Create directories if they do not exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    input_filename = os.path.join(input_dir, f"input_{fingerprint}.h5")
    output_filename = os.path.join(output_dir, f"output_{fingerprint}.h5")

    # Save input-only archive
    G.archive_inputonly(input_filename)

    # Extract output data
    power = np.array(G.output.field.power)
    energy = np.array(G.output.field.energy)
    energy_spectrum, spectrum = get_spectrum(G)
    zplot = np.array(G.output.lattice.zplot)
    s = np.array(G.output.globals.s)
    inital_beamenergy=np.array(G.output.beam.energy[0,:])
    inital_beamcurrent=np.array(G.output.beam.current[0])

    Lattice=G.output.lattice.aw*np.sqrt(2)

    # Save output to HDF5
    with h5py.File(output_filename, "w") as hdf5_file:
        hdf5_file.create_dataset("power", data=power)
        hdf5_file.create_dataset("energy", data=energy)
        hdf5_file.create_dataset("energy_spectrum", data=energy_spectrum)
        hdf5_file.create_dataset("spectrum", data=spectrum)
        hdf5_file.create_dataset("zplot", data=zplot)
        hdf5_file.create_dataset("s", data=s)
        hdf5_file.create_dataset("electron_beam_energy", data=inital_beamenergy)
        hdf5_file.create_dataset("current", data=inital_beamcurrent)
        hdf5_file.create_dataset("Lattice", data=Lattice)

    print(f"Archived input to: {input_filename}")
    print(f"Archived output to: {output_filename}")

    return fingerprint






def spectrum_from_field(field, dt=1):
    """
    Calculates the spectrum (fourier transformed field)
    from a complex field array with spacing dt.

    Parameters
    ----------
    field: nd.array of shape (n,)
        Complex field

    dt: float
        Spacing of the field data in some units (e.g. 's')

    Returns
    -------
    freqs: nd.array of shape (n,)
        Frequencies in reciprocal space with inverse units (e.g. 'Hz = 1/s')

    spectrum: nd.array of shape (n,)
        The fourier transformed field

    """
    if len(field.shape) != 1:
        raise ValueError("Only 1D arrays are currently supported")
    spectrum = np.fft.fftshift(np.fft.fft(field)) * dt

    ns = len(field)
    freqs = np.fft.fftshift(np.fft.fftfreq(ns, dt))

    return freqs, spectrum
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def calculate_spectrum(G):
    """
    Calculate the spectrum from the Genesis output object G and find the
    peak spectral fluence, the photon energy at the peak, and the FWHM.

    Parameters:
    G (Genesis object): The Genesis object containing simulation output data.

    Returns:
    peak_fluence (float): The maximum spectral fluence.
    peak_energy (float): The photon energy at the peak spectral fluence.
    fwhm_bandwidth (float): The Full Width at Half Maximum (FWHM) of the spectrum in eV.
    """
    print('getting output')
    output = G.output
    print('gotoutput')
    # Extract intensity, phase, and frequency information
    sig = output.field.intensity_farfield[-1, :]
    phi = output.field.phase_farfield[-1, :]
    freq = output.globals.frequency

    # Extract energy information
    energy = output.beam.energy[0, :] * 0.511 * 1e-3
    energy0 = energy[-1]

    # Calculate the signal and spectrum
    signal = np.sqrt(sig) * np.exp(1j * phi)
    spec = np.abs(np.fft.fftshift(np.fft.fft(signal))) ** 2
    norm = energy0 / np.sum(spec) / (freq[1] - freq[0])
    spec = norm * spec

 
    # Convert frequency to photon energy (eV) and compute spectral fluence (J/m^2/eV)
    photon_energies = h * freq  # eV
    spectral_fluence = spec  # Spectral fluence directly from normalized spectrum

    # Analyze spectrum to find peak, peak energy, and FWHM
    peak_fluence, peak_energy, fwhm_bandwidth = analyze_spectrum(freq, spectral_fluence)
    #plt.plot(freq, spec)
    #plt.plot(freq, gaussian(freq, peak_fluence, peak_energy, fwhm_bandwidth/(2 * np.sqrt(2 * np.log(2)))))
    #print(fwhm_bandwidth, 'FWHM')
    return peak_fluence, peak_energy, fwhm_bandwidth

def analyze_spectrum(photon_energies, spectral_fluence):
    """
    Analyze the spectrum to find the peak spectral fluence, 
    the photon energy at this peak, and the Full Width at Half Maximum (FWHM).

    Parameters:
    photon_energies (array-like): Array of photon energies in eV.
    spectral_fluence (array-like): Array of spectral fluence values corresponding to the photon energies.

    Returns:
    peak_fluence (float): The maximum spectral fluence.
    peak_energy (float): The photon energy at the peak spectral fluence.
    fwhm_bandwidth (float): The Full Width at Half Maximum (FWHM) of the spectrum in eV.
    """
    # Preserve the original maximum value for unnormalizing
    original_max = np.max(spectral_fluence)

    # Normalize the spectrum
    spectral_fluence = spectral_fluence / original_max

    # Define Gaussian function
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Initial guess for the Gaussian parameters [Amplitude, Mean, StdDev]
    initial_guess = [1, np.mean(photon_energies), np.std(photon_energies)]

    try:
         # Fit the Gaussian
        popt, _ = curve_fit(gaussian, photon_energies, spectral_fluence, p0=initial_guess)

        # Extract fitted parameters
        A_fit, mu_fit, sigma_fit = popt

        # Calculate FWHM
        fwhm_bandwidth = 2 * np.sqrt(2 * np.log(2)) * sigma_fit

        # Peak fluence and energy
        peak_fluence = A_fit * original_max  # Unnormalize the peak fluence
        peak_energy = mu_fit

    except Exception as e:
        # If fitting fails, set all variables to None
        A_fit = 'None'
        mu_fit = 'None'
        sigma_fit = 'None'
        fwhm_bandwidth = 'None'
        peak_fluence = 'None'
        peak_energy = 'None'
        print(f"Error in curve fitting: {e}")


    return peak_fluence, peak_energy, fwhm_bandwidth

def getPeakIntensity(G):
    output = G.output
    intensity=output.field.intensity_farfield[-1, :] * 1e-24
    intensity_max=np.max(intensity)
    return(intensity_max)

def merrit_genesis(G):
    # Check for error

    m={}

    #if G.output['run_info']['error']:
    #    return {'error':True}
    #else:
    #    m= {'error':False}
        
    m['energy']=G.stat('field_energy')[-1] *1e6

    peak_fluence, peak_energy, fwhm_bandwidth=calculate_spectrum(G)

    print('spectrum', peak_fluence, peak_energy, fwhm_bandwidth)
    
    m['bandwidth_FWHM']=fwhm_bandwidth

    m['peak_fluence']=peak_fluence

    m['peak_energy']=peak_energy

    m['peak_intensity']= getPeakIntensity(G)
    
    return m

def full_path(path):
    """
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))




def process_particle_group(file_path, alpha_x, beta_x, alpha_y, beta_y, twiss_match=True):
    """
    Loads a ParticleGroup from a file, resamples it, drifts it to z-coordinates,
    and applies Twiss matching if enabled.

    Parameters
    ----------
    file_path : str
        Path to the input HDF5 particle file.
    alpha_x : float
        Twiss alpha parameter for the x-plane.
    beta_x : float
        Twiss beta parameter for the x-plane.
    alpha_y : float
        Twiss alpha parameter for the y-plane.
    beta_y : float
        Twiss beta parameter for the y-plane.
    twiss_match : bool, optional
        Whether to apply Twiss matching (default is True).

    Returns
    -------
    ParticleGroup
        The processed ParticleGroup object.
    """
    # Load the particle group from file
    P1 = ParticleGroup(file_path)

    # Resample particles to maintain distribution
    NSAMPLE = len(P1)
    P1 = P1.resample(NSAMPLE)

    # Drift particles to z-coordinates
    P1.drift_to_z()

    # Apply Twiss matching if enabled
    if twiss_match:
        P1.twiss_match(beta=beta_x, alpha=alpha_x, plane='x', inplace=True)
        P1.twiss_match(beta=beta_y, alpha=alpha_y, plane='y', inplace=True)

    return P1  # Return the processed ParticleGroup







def runGenesis(input_dict, run_file_location, run_file,Lattice_file, workdir, nproc=2):



    beam_length= input_dict.get('beam_length', 15e-6)
    curr0= input_dict.get('current', 3000)
    gamma0=input_dict.get('gamma0', 19174.0776) 
    chirp = input_dict.get('chirp', -15)

    slen=20e-6
    
    file=run_file_location + run_file
    
    G = Genesis4(run_file_location + run_file, lattice= run_file_location + Lattice_file, workdir=workdir, verbose = False)


    if 'alphax' in input_dict:
        G.input.main.beam.alphax = input_dict["alphax"] #alpha_x
        alpha_x=input_dict["alphax"] 
    else: 
        alpha_x=G.input.main.beam.alphax

    if "alphay" in input_dict:
        G.input.main.beam.alphay = input_dict["alphay"]
        alpha_y=input_dict["alphay"]
    else:
        alpha_y=G.input.main.beam.alphay
        
    if "betax" in input_dict:
        G.input.main.beam.betax= input_dict["betax"]
        beta_x=input_dict["betax"]
    else:
        beta_x= G.input.main.beam.betax
        
    if "betay" in input_dict:
        G.input.main.beam.betay = input_dict["betay"]
        beta_y=input_dict["betay"]
    else:
        beta_y= G.input.main.beam.betay

    if "emitx" in input_dict:
        G.input.main.beam.ex = input_dict["emitx"]
        emit_x=input_dict["emitx"]
    else:
        emit_x= G.input.main.beam.ex

    if "emity" in input_dict:
        G.input.main.beam.ey = input_dict["emity"]
        emit_y=input_dict["emity"]
    else:
        emit_y= G.input.main.beam.ey
        
    if 'taper' in input_dict or 'taper_quadratic' in input_dict:

            print('in taper')

            dKbyK_linear = input_dict.get('taper', 0)  #input_dict['taper']  # Use provided taper value for linear taper
 
            # Check for separate quadratic taper
            dKbyK_quadratic = input_dict.get('taper_quadratic', 0)
 
            # Undulator section indices
            ustart = input_dict.get('ustart', 0)
            ustart =int(ustart)
            u_linear_stop = input_dict.get('u_linear_stop', 12)  # Default transition point
            u_linear_stop= int(u_linear_stop)
            u_quadratic_stop = input_dict.get('u_quadratic_stop', 33)  # Default end of quadratic taper
            u_quadratic_stop = int(u_quadratic_stop)
   
            # Undulator properties
            Kstart = 1.7017
            nwig = 130
            xlamdu = 0.026
  
            # Apply both linear and quadratic tapering
            G.apply_taper(
                Kstart=Kstart, 
                dKbyK_linear=dKbyK_linear, 
                dKbyK_quadratic=dKbyK_quadratic, 
                ustart=ustart, 
                u_linear_stop=u_linear_stop, 
                u_quadratic_stop=u_quadratic_stop, 
                nwig=nwig, 
                uperiod=xlamdu
            ) 

    beam_fingerprint = generate_beam_fingerprint(input_dict)
                                                

    G.input.main.time.sample=80
   
    G.input.main.namelists.append(Write(field="end"))
    G.input.main.track.zstop=500
    G.input.main.time.slen=slen


    gamma_file_path, current_file_path=generate_beam_profiles(slen, curr0, gamma0, chirp, beam_length, target_dir= workdir, fingerprint=beam_fingerprint) 
  
    basename_gamma = os.path.basename(gamma_file_path)
    basename_current = os.path.basename(current_file_path)
    
    label_gamma = os.path.splitext(basename_gamma)[0]     
    label_current = os.path.splitext(basename_current)[0]  

    # Add to Genesis input 
    G.input.main.namelists.insert(5,ProfileFile(
        label=label_gamma,
        xdata=f"{basename_gamma}/s",
        ydata=f"{basename_gamma}/gamma"
    ))

    # Add to Genesis input 
    G.input.main.namelists.insert(5,ProfileFile(
        label=label_current,
        xdata=f"{basename_current}/s",
        ydata=f"{basename_current}/current"
    ))
    
    
    G.input.main.beam.gamma='@'+label_gamma
    G.input.main.beam.current='@'+label_current
    
    G.write_input(path=G.path, write_run_script=True)
        
 

    #shutil.move(gamma_filename, G.path + "/" + 'beam_gamma.h5')
    #shutil.move(current_filename, G.path + "/" + "beam_current.h5")
    

    #basename = os.path.basename(gamma_filename)  # -> 'beam_gamma_-15.000.h5'
    #label = os.path.splitext(basename)[0]        # -> 'beam_gamma_-15.000'

    # Add to Genesis input 
   # G.input.main.namelists.append(ProfileFile(
   #     label=label,
   #     xdata=f"{basename}/s",
   #     ydata=f"{basename}/gamma"
   # ))



    print(nproc)
    G.nproc = nproc
    # SDF MPI setup with time limit

  #  print('here')
    
    #print('adding mpi')
    print(G.mpi_run)
    #G.mpi_run = 'salloc --partition milano --account ad:beamphysics --mem-per-cpu=4g  --time=0-00:15:00 -n {nproc} mpirun -n {nproc} {command_mpi}'
#    G.mpi_run = 'salloc --partition roma --account ad:beamphysics --mem-per-cpu=4g  -n {nproc} mpirun -n {nproc} {command_mpi}'
    G.mpi_run = (
        'salloc --partition roma '
        '--account ad:beamphysics '
        '--mem-per-cpu=4g '
        '--cpus-per-task=1 '
        ' -n {nproc} '
       'mpirun -n {nproc} {command_mpi}' 
    )

    G.mpi_run = (
         'salloc --partition roma '
         '--account ad:beamphysics '
         '--mem-per-cpu=4g '
         '--ntasks={nproc} '
         '--cpus-per-task=1 '
         '--time=00:10:00 '  # <-- FIXED
         'mpirun --bind-to none -np {nproc} {command_mpi}')


    print(G.mpi_run, 'mpi')
#    G.mpi_run = f'salloc --partition milano --account ad:beamphysics --mem-per-cpu=4g -n {nproc} mpirun -n {nproc} {command_mpi}'

   # G.mpi_run='salloc --partition milano --account ad:beamphysics --mem-per-cpu=4g -n {nproc} mpirun -n {nproc} {command_mpi}'
   # G.mpi_run = 'salloc --partition milano --account ad:beamphysics --mem-per-cpu=4g -n {nproc} mpirun --cpu-bind=core,overload-allowed -n {nproc} {command_mpi}'
    
    #print('now run?')



    G.run()
    #print('runc done')    
    twiss=[ beam_fingerprint] #[alpha_x, alpha_y, beta_x, beta_y , emit_x, emit_y, ]



    if os.path.exists(gamma_file_path):
       os.remove(gamma_file_path)
       print(f"Deleted: {gamma_file_path}")
    else:
        print(f"File not found: {gamma_file_path}")

    if os.path.exists(current_file_path):
       os.remove(current_file_path)
       print(f"Deleted: {current_file_path}")
    else:
        print(f"File not found: {current_file_path}")
 
    return G, twiss


def evaluate_genesis(input_dict, run_file_location, run_file, Lattice_file,  workdir, nproc=2):
    """
     If an archive_path is given, the complete evaluated Genesiss and Generator objects will be archived
     to a file named using a fingerprint from both objects.
    """
    input_dict = dict(input_dict) 
    d={}
    print('input dictonary', input_dict)
    G , twiss = runGenesis(input_dict=input_dict, run_file_location=run_file_location, run_file=run_file, Lattice_file=Lattice_file ,workdir=workdir, nproc=nproc)
    
    # Evaluate merit of Genesis output
    output=merrit_genesis(G)  

    d['inputs'] = [input_dict]

#    d['twiss']=twiss

 #   if 'error' in output and output['error']:

  #       raise ValueError('run_genesis returned error in output')
   
        
    #Recreate Generator object for fingerprint, proper archiving
    # TODO: make this cleaner
    #Gen = Generator(G.input)  ## don't need this unless distgen file is used. 


    fingerprint=archive_with_fingerprint(G)

    print(fingerprint, 'fingerprint')

    
    
    output['fingerprint'] = fingerprint

    output['chirp_used'] = twiss
    d['outputs'] = output

    
    return output
