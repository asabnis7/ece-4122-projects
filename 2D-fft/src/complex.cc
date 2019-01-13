//
// Created by brian on 11/20/18.
//

#include "complex.h"

#include <cmath>

const float PI = 3.14159265358979f;

Complex::Complex() : real(0.0f), imag(0.0f) {}

Complex::Complex(float r) : real(r), imag(0.0f) {}

Complex::Complex(float r, float i) : real(r), imag(i) {}

Complex Complex::operator+(const Complex &b) const {
    Complex temp;
    temp.real = this->real + b.real;
    temp.imag = this->imag + b.imag;
    return temp;
}

Complex Complex::operator-(const Complex &b) const {
    Complex temp;
    temp.real = this->real - b.real;
    temp.imag = this->imag - b.imag;
    return temp;
}

Complex Complex::operator*(const Complex &b) const {
    Complex temp;
    temp.real = (this->real)*b.real - (this->imag)*b.imag;
    temp.imag = (this->real)*b.imag + (this->imag)*b.real;
    return temp;
}

float Complex::mag() const {
    return sqrt(pow(this->real,2)+pow(this->imag,2));
}

float Complex::angle() const {
    return atan2(this->imag, this->real); 
}

Complex Complex::conj() const {
    Complex temp(this->real, -(this->imag));
    return temp;
}

std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}
