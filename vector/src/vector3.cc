#include <iostream>
#include "vector3.h"

using namespace std;
// xyz constructor
Vector3::Vector3(float xyz)
{
    x = xyz;
    y = xyz;
    z = xyz; 
}

// component-wise constructor
Vector3::Vector3(float _x, float _y, float _z): x(_x), y(_y), z(_z) {}
// component-wise operators +-*/
Vector3 Vector3::operator+(const Vector3& rhs)
{
    float _x, _y, _z;
    _x = this->x + rhs.x;
    _y = this->y + rhs.y;
    _z = this->z + rhs.z;
    return Vector3(_x,_y,_z);
}

Vector3 Vector3::operator-(const Vector3& rhs)
{
    float _x, _y, _z;
    _x = this->x - rhs.x;
    _y = this->y - rhs.y;
    _z = this->z - rhs.z;
    return Vector3(_x,_y,_z);
}

Vector3 Vector3::operator*(const Vector3& rhs)
{
    float _x, _y, _z;
    _x = this->x * rhs.x;
    _y = this->y * rhs.y;
    _z = this->z * rhs.z;
    return Vector3(_x,_y,_z);
}

Vector3 Vector3::operator/(const Vector3& rhs)
{
    float _x, _y, _z;
    _x = this->x / rhs.x;
    _y = this->y / rhs.y;
    _z = this->z / rhs.z;
    return Vector3(_x,_y,_z);
}

// scalar operators +-*/
Vector3 Vector3::operator+(float rhs)
{
    float _x, _y, _z;
    _x = this->x + rhs;
    _y = this->y + rhs;
    _z = this->z + rhs;
    return Vector3(_x, _y, _z);
}

Vector3 Vector3::operator-(float rhs)
{
    float _x, _y, _z;
    _x = this->x - rhs;
    _y = this->y - rhs;
    _z = this->z - rhs;
    return Vector3(_x, _y, _z);
}

Vector3 Vector3::operator*(float rhs)
{
    float _x, _y, _z;
    _x = this->x * rhs;
    _y = this->y * rhs;
    _z = this->z * rhs;
    return Vector3(_x, _y, _z);
}

Vector3 Vector3::operator/(float rhs)
{
    float _x, _y, _z;
    _x = this->x / rhs;
    _y = this->y / rhs;
    _z = this->z / rhs;
    return Vector3(_x, _y, _z);
}

// dot product
float Vector3::operator|(const Vector3& rhs)
{
    float result = (this->x)*rhs.x + (this->y)*rhs.y + (this->z)*rhs.z;
    return result;
}

// cross product
Vector3 Vector3::operator^(const Vector3& rhs)
{
    float _x, _y, _z;
    _x = ((this->y)*rhs.z)-((this->z)*rhs.y); 
    _y = -1.0*(((this->x)*rhs.z)-((this->z)*rhs.x));
    _z = ((this->x)*rhs.y)-((this->y)*rhs.x);
    return Vector3(_x, _y, _z);
}

// component-wise operation-assignment operators += -= *= /=
Vector3& Vector3::operator+=(const Vector3& rhs)
{
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
}

Vector3& Vector3::operator-=(const Vector3& rhs)
{
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    return *this;
}

Vector3& Vector3::operator*=(const Vector3& rhs)
{
    this->x *= rhs.x;
    this->y *= rhs.y;
    this->z *= rhs.z;
    return *this;
}

Vector3& Vector3::operator/=(const Vector3& rhs)
{
    this->x /= rhs.x;
    this->y /= rhs.y;
    this->z /= rhs.z;
    return *this;
}

// pre/post-increment right rotate operators
Vector3& Vector3::operator++()
{
    float _x = this->x;
    float _y = this->y;
    float _z = this->z;
    
    this->x = _z;
    this->y = _x;
    this->z = _y;

    return *this;
}

Vector3 Vector3::operator++(int __unused)
{
    Vector3 temp(this->x, this->y, this->z);
    operator++();
    return temp;
}

// pre/post-decrement left rotate operators
Vector3& Vector3::operator--()
{
    float _x = this->x;
    float _y = this->y;
    float _z = this->z;

    this->x = _y;
    this->y = _z;
    this->z = _x;

    return *this;
}

Vector3 Vector3::operator--(int __unused)
{
    Vector3 temp(this->x, this->y, this->z);
    operator--();
    return temp;
}

// equality operators
bool Vector3::operator==(const Vector3& rhs)
{
    bool _x, _y, _z;
    _x = (this->x) == rhs.x;    
    _y = (this->y) == rhs.y;
    _z = (this->z) == rhs.z;

    return (_x && _y && _z);
}

// inequality operator
bool Vector3::operator!=(const Vector3& rhs)
{
    bool _x, _y, _z;
    _x = (this->x) != rhs.x;    
    _y = (this->y) != rhs.y;
    _z = (this->z) != rhs.z;

    return (_x && _y && _z);
}

// debugging purposes
void Vector3::print()
{
    cout << "x = " << x << ", y = " << y << ", z = " << z << endl;
}
