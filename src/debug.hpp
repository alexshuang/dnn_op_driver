#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

#ifdef DEBUG
#define debug(fmt, ...)	printf(fmt, __VA_ARGS__)
#else
#define debug(fmt, ...)
#endif

#endif
