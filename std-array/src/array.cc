// Arjun Sabnis
// ECE 4122 - Fall 2018
// Dynamic array template class

#include "array.h"
#include "simple_string.h"

// array class implementation ---------------------------------------------------------------------

// default constructor
template<typename T>
array<T>::array(): 
    m_elements(nullptr), 
    m_size(0), 
    m_reserved_size(0) 
    {
//        std::cout << "Default constructor used" << std::endl;
    } 

// initializer list constructor
template<typename T>
array<T>::array(std::initializer_list<T> vals):
    m_size(vals.size()),
    m_reserved_size(vals.size()),
    m_elements((T*) malloc(vals.size()*sizeof(T)))
    {
        for (int i = 0; i < vals.size(); i++) 
            new (&m_elements[i]) T(*(vals.begin()+i)); 
//        std::cout << "Initializer list constructor with " << m_size << " elements" << std::endl;
    }

// copy constructor
template<typename T>
array<T>::array(const array& copy):
    m_size(copy.m_size),
    m_reserved_size(copy.m_reserved_size),
    m_elements((T*) malloc(m_size*sizeof(T)))
    {
        if (m_elements != nullptr) 
            std::copy(copy.m_elements, copy.m_elements + m_size, m_elements);
//        std::cout << "Copy constructor used" << std::endl;    
    }

// move constructor
template<typename T>
array<T>::array(array&& move):
    m_size(move.m_size),
    m_reserved_size(move.m_reserved_size),
    m_elements(move.m_elements)
    {
        move.m_elements = nullptr;
        move.m_size = 0;
        move.m_reserved_size = 0;
//        std::cout << "Move constructor used" << std::endl;
    }

// constructor with initial reserved size
template<typename T>
array<T>::array(size_t reserved):
    m_size(0),
    m_reserved_size(reserved),
    m_elements((T*) malloc(reserved*sizeof(T)))
    {
//        std::cout << "Array constructed with " << reserved << " reserved elements" << std::endl;
    }  

// constructor with n copies of T
template<typename T>
array<T>::array(size_t n, const T& t):
    m_size(n),
    m_reserved_size(n)
    {
        m_elements = (T*) malloc(n*sizeof(T));
        for (int i = 0; i < n; i++)
            new (&m_elements[i]) T(t);
//        std::cout << "Array constructed with " << n << " copies of element" << std::endl;
    }

// class destructor
template<typename T>
array<T>::~array()
{
    if(m_elements)
    {
        for (int i = 0; i < m_size; i++)
            m_elements[i].~T();
    }
    free(m_elements); 
    m_elements = nullptr;
    m_size = 0;
    m_reserved_size = 0;
}

// reserve memory for size n
template<typename T>
void array<T>::reserve(size_t n)
{   
    this->m_reserved_size = n;
    
    if (m_elements != nullptr) 
    {
        T* temp = (T*) malloc(m_reserved_size*sizeof(T));
        for (int i = 0; i < m_size; i++)
        {
            new (&temp[i]) T(std::move(m_elements[i]));
            m_elements[i].~T();
        }
        free(m_elements);
        m_elements = temp;
    }

    else if (m_elements == nullptr) m_elements = (T*) malloc(m_reserved_size*sizeof(T)); 

//    std::cout << m_reserved_size << " elements reserved, array reallocated" << std::endl;
}

// push back element
template<typename T>
void array<T>::push_back(const T& val)
{
    if ((m_size == m_reserved_size) && m_reserved_size > 0) reserve(2*m_size);
    else if (m_reserved_size == 0) reserve(1);  
    new (&m_elements[m_size]) T(std::move(val));
    this->m_size++;
}

// push front element
template<typename T>
void array<T>::push_front(const T& val)
{
    if (m_reserved_size == 0) reserve(1);
    else if ((m_size == m_reserved_size) && m_reserved_size > 0) reserve(2*m_size);
    
    T* temp = (T*) malloc(m_reserved_size*sizeof(T));
    for (int i = 0; i < m_size; i++)
    {
        new (&temp[i+1]) T(std::move(m_elements[i]));
        m_elements[i].~T();
    }
    free(m_elements);
    m_elements = temp; 
 
    new (&m_elements[0]) T(std::move(val));
    this->m_size++;
}

// pop back element
template<typename T>
void array<T>::pop_back()
{
    if (!empty())
    {
        this->m_size--;
        m_elements[m_size].~T();
    }
    else std::cout << "Array is already empty." << std::endl;
}

// pop front element
template<typename T>
void array<T>::pop_front()
{
    if (!empty())
    {
        m_elements[0].~T();
        std::move(m_elements + 1, m_elements + m_size, m_elements);
        this->m_size--;
    }
    else std::cout << "Array is already empty." << std::endl;
}

// reference to first element
template<typename T>
T& array<T>::front() const
{
    return m_elements[0]; 
}

// reference to back element
template<typename T>
T& array<T>::back() const
{ 
    return m_elements[m_size-1];
}

// const reference to specific element
template<typename T>
const T& array<T>::operator[](size_t i) const
{
    return m_elements[i];
}

// non-const reference to specific element
template<typename T>
T& array<T>::operator[](size_t i)
{
    return m_elements[i];
}

// number of elements in array
template<typename T>
size_t array<T>::length() const
{
    return m_size;
}

// bool for emptiness
template<typename T>
bool array<T>::empty() const
{
    if (m_size == 0) return true;
    else return false;
}

// clear all elements
template<typename T>
void array<T>::clear()
{
    if(m_elements)
    {
        for (int i = 0; i < m_size; i++)
            m_elements[i].~T();
    }
    m_size = 0;
//    std::cout << "Array cleared. All elements removed" << std::endl;
}

// obtain iterator to first element
template<typename T>
array_iterator<T> array<T>::begin() const
{
    if (m_size != 0) return array_iterator<T>(m_elements);
    else
    {
        std::cout << "Array is empty. Returning null iterator" << std::endl;
        return array_iterator<T>(nullptr);
    }
}

// obtain one-past end iterator
template<typename T>
array_iterator<T> array<T>::end() const
{
    if (m_size != 0) return array_iterator<T>(m_elements + m_size);
    else
    {
        std::cout << "Array is empty. Returning null iterator." << std::endl;
        return array_iterator<T>(nullptr);
    }
}

// remove specified element
template<typename T>
void array<T>::erase(const array_iterator<T>& pos)
{
    if (pos == begin()) pop_front();
    else if (pos == end()) pop_back();

    else
    {
        array_iterator<T> it;
        int position = 0;
        for (it = m_elements; it != m_elements + m_size; it++)
        {
            if (it == pos) break;
            position++;
        }
        m_elements[position].~T(); // delete element
        std::move(m_elements+(position+1), m_elements+m_size, m_elements+(position));
        this->m_size--;
    }
}

// insert new element
template<typename T>
void array<T>::insert(const T& value, const array_iterator<T>& pos)
{ 
    if (this->m_reserved_size == 0) push_back(value);
    else if (pos == begin()) push_front(value);
    else if (pos == end()) push_back(value);
    else
    { 
        array_iterator<T> it;
        int position = 0;
        for (it = m_elements; it != m_elements + m_size; it++)
        {
            if (it == pos) break;
            position++;
        } 
        if (m_size < m_reserved_size)
        { 
            new (&m_elements[m_size]) T(std::move(m_elements[m_size-1]));
            std::move(m_elements+position, m_elements+(m_size), m_elements+(position+1));
            m_elements[position].~T();
            new (&m_elements[position]) T(value);
        }
        else if (m_size == m_reserved_size) 
        {
            this->m_reserved_size *= 2;
            T* temp = (T*) malloc(this->m_reserved_size*sizeof(T));
            new (&temp[position]) T(value);
            for (int i = 0; i < position; i++)
            {
                new (&temp[i]) T(std::move(m_elements[i]));
                m_elements[i].~T();
            }
            for (int i = position; i < m_size; i++)
            {
                new (&temp[i+1]) T(std::move(m_elements[i]));
                m_elements[i].~T();
            }
            free(m_elements);
            m_elements = temp;
        }
        this->m_size++;
    }
}


// print array for debugging
template<typename T>
void array<T>::print()
{
    for(int i = 0; i < m_size; i++) std::cout << " " << m_elements[i] << " ";
    std::cout << std::endl;
}

// end array class implementation -----------------------------------------------------------------




// array iterator class implementation ------------------------------------------------------------

// default iterator constructor
template<typename T>
array_iterator<T>::array_iterator():
    m_current(nullptr) {}

// iterator constructor with address
template<typename T>
array_iterator<T>::array_iterator(T* ptr):
    m_current(ptr) {}

// copy constructor
template<typename T>
array_iterator<T>::array_iterator(const array_iterator& it):
    m_current(it.m_current) 
{
    //std::cout << "Array iterator copy constructor used" << std::endl;
}

// iterator dereference
template<typename T>
T& array_iterator<T>::operator*() const
{ 
    return *m_current;
}

// predecrement iterator
template<typename T>
array_iterator<T> array_iterator<T>::operator++()
{
    this->m_current++;
    return *this;
}

// postdecrement iterator
template<typename T>
array_iterator<T> array_iterator<T>::operator++(int __unused)
{
    array_iterator temp(this->m_current);
    this->m_current++;
    return temp;
}

// inequality operator
template<typename T>
bool array_iterator<T>::operator!=(const array_iterator& rhs) const
{
    return (this->m_current != rhs.m_current);
}

// equality operator
template<typename T>
bool array_iterator<T>::operator==(const array_iterator& rhs) const
{
    return (this->m_current == rhs.m_current);
}

// end array iterator class implementation --------------------------------------------------------


// avoid linker issues during compile
template class array<int>;
template class array<double>;
template class array<char>;
template class array<float>;
template class array<int*>;
template class array<double*>;
template class array<char*>;
template class array<float*>;
template class array<simple_string>;
template class array<simple_string*>;

template class array_iterator<int>;
template class array_iterator<double>;
template class array_iterator<char>;
template class array_iterator<float>;
template class array_iterator<int*>;
template class array_iterator<double*>;
template class array_iterator<char*>;
template class array_iterator<float*>;
template class array_iterator<simple_string>;
template class array_iterator<simple_string*>;
