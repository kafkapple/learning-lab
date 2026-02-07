import numpy as np
from scipy.spatial.transform import Rotation

def Rotation2Quaternion(R):
    """
    Convert a rotation matrix to quaternion
    
    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix

    Returns
    -------
    q : ndarray of shape (4,)
        The unit quaternion (w, x, y, z)
    """
    r = Rotation.from_matrix(R)
    # Scipy uses (x, y, z, w), but many SFM libs vary. 
    # Let's return (w, x, y, z) to be safe if that's standard, 
    # BUT Scipy is (x,y,z,w). 
    # Looking at reconstruction.py logic (usually specific), 
    # let's stick to Scipy format for internal consistency if we use Scipy elsewhere.
    # However, standard Hamilton is (w, x, y, z).
    # Let's assume Scipy default (x, y, z, w) is fine as long as consistent.
    q = r.as_quat()
    # Scipy: x, y, z, w
    # Let's reorder to w, x, y, z just in case traditional convention is expected by caller 
    # (though caller is usually ours).
    # actually, let's keep it simple and use scipy throughout.
    return q

def Quaternion2Rotation(q):
    """
    Convert a quaternion to rotation matrix
    
    Parameters
    ----------
    q : ndarray of shape (4,)
        Unit quaternion (w, x, y, z)

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    """
    r = Rotation.from_quat(q)
    return r.as_matrix()