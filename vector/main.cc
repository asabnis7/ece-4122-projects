#include <iostream>
#include "vector3.h"

using namespace std;

int main()
{
    Vector3 v1(12.0);
    Vector3 v2(-3.0, -2.8, -5.6);
    Vector3 v3(2.7, 3.2, 4.5);

    cout << "v1: "; v1.print();
    cout << "v2: "; v2.print();
    cout << "v3: "; v3.print();
    cout << endl;

    cout << "component-wise operator tests" << endl;
    // v+v
    cout << "v2 = v2 + v1" << endl;
    v2 = v2 + v1;
    cout << "v1: "; v1.print();
    cout << "v2: "; v2.print();
    cout << endl;

    // v-v
    cout << "v3 = v1 - v3" << endl;
    v3 = v1 - v3;
    cout << "v1: "; v1.print();
    cout << "v3: "; v3.print();
    cout << endl;

    // v*v
    cout << "v1 = v2 * v2" << endl;
    v1 = v2*v2;
    cout << "v1: "; v1.print();
    cout << "v2: "; v2.print();
    cout << endl;

    // v/v
    cout << "v3 = v1 / v2" << endl;
    v3 = v1/v2;
    cout << "v1: "; v1.print();
    cout << "v2: "; v2.print();
    cout << "v3: "; v3.print();
    cout << endl;

    cout << "scalar operation tests" << endl;
    // v+num
    cout << "v2 = v2 + 4.0" << endl;
    v2 = v2 + 4.0;
    cout << "v2: "; v2.print();
    cout << endl;

    // v-num
    cout << "v3 = v1 - 9.9" << endl;
    v3 = v1 - 9.9;
    cout << "v1: "; v1.print();
    cout << "v3: "; v3.print();
    cout << endl;

    // v*num
    cout << "v1 = v2 * 2.0" << endl;
    v1 = v2*2.0;
    cout << "v1: "; v1.print();
    cout << "v2: "; v2.print();
    cout << endl;

    // v/num
    cout << "v3 = v1 / 5.0" << endl;
    v3 = v3/5.0;
    cout << "v1: "; v1.print();
    cout << "v3: "; v3.print();
    cout << endl;

    cout << "dot product test: v1|v2" << endl;
    float result = v1|v2;
    cout << "result = " << result << endl;
    cout << "v1: "; v1.print();
    cout << "v2: "; v2.print();
    cout << endl;

    cout << "cross product test: v2 = v1^v3" << endl;
    v2 = v1^v3;
    cout << "v1: "; v1.print();
    cout << "v2: "; v2.print();
    cout << "v3: "; v3.print();
    cout << endl;

    cout << "component-wise assignment operator tests" << endl;
    // v+=v
    cout << "v2 += v1" << endl;
    v2 += v1;
    cout << "v1: "; v1.print();
    cout << "v2: "; v2.print();
    cout << endl;

    // v-=v
    cout << "v3 -= v1" << endl;
    v3 -= v1;
    cout << "v1: "; v1.print();
    cout << "v3: "; v3.print();
    cout << endl;

    // v*=v
    cout << "v1 *= v2" << endl;
    v1 *= v2;
    cout << "v1: "; v1.print();
    cout << "v2: "; v2.print();
    cout << endl;

    // v/=v
    cout << "v3 /= v2" << endl;
    v3 /= v2;
    cout << "v2: "; v2.print();
    cout << "v3: "; v3.print();
    cout << endl;

    // ++v; v++
    cout << "increment tests" << endl;    
    cout << "v2: "; v2.print();
    cout << "preincrement" << endl;
    (++v2).print();
    cout << "v2: "; v2.print();
    cout << "before postincrement" << endl;
    (v2++).print();
    cout << "after postincrement" << endl;
    cout << "v2: "; v2.print();
    cout << endl;
    
    // --v; v--
    cout << "decrement tests" << endl;    
    cout << "v1: "; v1.print();
    cout << "predecrement" << endl;
    (--v1).print();
    cout << "v1: "; v1.print();
    cout << "before postdecrement" << endl;
    (v1--).print();
    cout << "after postdecrement" << endl;
    cout << "v1: "; v1.print();
    cout << endl;

    // equality operator tests
    Vector3 v4(1.0);
    Vector3 v5(1.0);
    Vector3 v6(4.0);

    cout << "v4: "; v4.print();
    cout << "v5: "; v5.print();
    cout << "v6: "; v6.print();
    cout << endl;

    cout << "v4 == v5: " << (v4 == v5) << endl;
    cout << "v4 == v6: " << (v4 == v6) << endl;
    cout << "v4 != v5: " << (v4 != v5) << endl;
    cout << "v4 != v6: " << (v4 != v6) << endl;
}
