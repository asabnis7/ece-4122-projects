// Arjun Sabnis
// ECE 4122 - Fall 2018
// 3D Vector struct

struct Vector3 {
    
    float x;
    float y;
    float z;

    // Constructors
    Vector3() = default;
    Vector3(float xyz);
    Vector3(float x, float y, float z);

    // Component-wise vector operations
    Vector3 operator+(const Vector3& rhs);
    Vector3 operator-(const Vector3& rhs);
    Vector3 operator*(const Vector3& rhs);
    Vector3 operator/(const Vector3& rhs);

    // Scalar operations
    Vector3 operator+(float rhs);
    Vector3 operator-(float rhs);
    Vector3 operator*(float rhs);
    Vector3 operator/(float rhs);

    //Dot product
    float operator|(const Vector3& rhs);
    //Cross product
    Vector3 operator^(const Vector3& rhs);

    // More component-wise vector operations
    Vector3& operator +=(const Vector3& rhs);
    Vector3& operator -=(const Vector3& rhs);
    Vector3& operator *=(const Vector3& rhs);
    Vector3& operator /=(const Vector3& rhs);

    // Pre/post-increment rotate Vector3 to right
    // x = z, y = x, z = y
    Vector3& operator++();
    Vector3 operator++(int __unused);

    // Pre/post-decrement rotate Vector3 to left
    // x = y, y = z, z = x
    Vector3& operator--();
    Vector3 operator--(int __unused);

    // Component-wise equality operators
    bool operator==(const Vector3& rhs);
    bool operator!=(const Vector3& rhs);

    // For debugging
    void print();
};
