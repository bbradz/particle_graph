import json
import os

def load_pdg_data(file_path='json2fr/pdg_sm.json'):
    """Load particle data from PDG JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: PDG data file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None

import hashlib

def generate_bsm_pdg_id(mass, spin, charge, color, offset=9000000):
    """
    Generate a unique BSM PDG ID >= 9000000 based on particle properties.

    Parameters:
        mass (float): Mass of the particle in GeV
        spin (int): Spin (0, 1, 2) / 2
        charge (int): Electric charge in units of e/3 (e.g., 3 for e⁻)
        color (int): Color representation: 1=singlet, 3=triplet, 8=octet
        offset (int): Minimum PDG ID to start from (default 9000000)

    Returns:
        int: A positive PDG ID ≥ offset
    """

    # Input check
    if color not in [1, 3, 8]:
        raise ValueError("Color must be 1 (singlet), 3 (triplet), or 8 (octet)")

    # Normalize and stringify inputs
    props = f"{mass:.6f}|{spin}|{charge}|{color}"

    # Hash the string to get a reproducible number
    hashed = hashlib.md5(props.encode()).hexdigest()

    # Convert part of the hash to an integer
    numeric_hash = int(hashed[:8], 16)  # Use 8 hex digits (32-bit)

    # Map to PDG ID space
    bsm_pdg_id = offset + (numeric_hash % (9999999 - offset))

    return bsm_pdg_id

def get_pdg_id(mass=None, 
               charge=None, 
               color=None, 
               spin=None,
               flavor=None,
               tolerance=0.1
               ):
    """
    Find PDG ID based on particle properties.
    
    Parameters:
    -----------
    mass : float, optional
        Particle mass in GeV
    charge : int, optional
        Electric charge (in units of e/3)
    color : int, optional
        Color representation (1 for singlet, 3 for triplet, 8 for octet)
    spin : int, optional
        Particle spin (0 for scalar, 1 for fermion, 1 for vector)
    name : str, optional
        Particle name for direct lookup
    tolerance : float, optional
        Relative tolerance for mass comparison
        
    Returns:
    --------
    int or None
        PDG ID if found, None otherwise
    """
    pdg_data = load_pdg_data()
    if not pdg_data:
        return None
    
    # Property-based lookup
    candidates = pdg_data['particles']
    
    # Filter by properties if provided
    if charge is not None:
        candidates = [p for p in candidates if p['charge'] == charge]
    
    if color is not None:
        candidates = [p for p in candidates if p['color'] == color]
    
    if spin is not None:
        candidates = [p for p in candidates if p['spin'] == spin]

    if mass is not None:
        candidates = [p for p in candidates if abs(p['mass'] - mass) <= tolerance * mass or (p['mass'] == mass)]

    if flavor is not None:
        candidates = [p for p in candidates if p['flavor'] == flavor]

    # Return the PDG ID if we have a unique match
    if len(candidates) == 1:
        return candidates[0]['pdgid']
    else:
        return generate_bsm_pdg_id(mass, spin, charge, color)


def get_particle_properties(pdg_id):
    """
    Get particle properties based on PDG ID.
    
    Parameters:
    -----------
    pdg_id : int
        PDG ID of the particle
        
    Returns:
    --------
    dict or None
        Dictionary of particle properties if found, None otherwise
    """
    pdg_data = load_pdg_data()
    if not pdg_data:
        return None
    
    for particle in pdg_data['particles']:
        if particle['pdgid'] == pdg_id:
            return particle
    
    print(f"No particle found with PDG ID {pdg_id}")
    return None

# Example usage
if __name__ == "__main__":
    # Find electron by properties
    electron_id = get_pdg_id(mass=0.000511, charge=-3, color=1, spin=1)
    print(f"Electron PDG ID: {electron_id}")