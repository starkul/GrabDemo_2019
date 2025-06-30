#pragma once
#include <cstdio>
#include <cstdint>

using namespace std;

using float16_t2 = uint16_t;

struct Float16
{
	float16_t2 _value;
	Float16(float16_t2 value)
	{
		_value = value;
	}
	Float16(float value)
	{
		uint32_t m = *(uint32_t *)&value;
		_value = ((m & 0x7fffffff) >> 13) - (0x38000000 >> 13);
		_value |= ((m & 0x80000000) >> 16);
	}
	operator float16_t2 ()
	{
		return _value;
	}
	operator float()
	{
		uint32_t m = _value;
		m = ((m & 0x7fff) << 13) + 0x38000000;
		m |= ((_value & 0x8000) << 16);
		return *(float *)&m;
	}
};