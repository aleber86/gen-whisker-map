
double modulus(double arg, double _fmod){
    /*
     * Floating point modulus operation. Allways positive value.
     *
     * arg : argument of function modulo
     * _fmod : congruent value
     *
     * returns (double) modulo
     */
    double _modulus = fmod(arg, _fmod);
    if (_modulus < 0){
        _modulus = _modulus + _fmod;
    }
    return _modulus;
}
