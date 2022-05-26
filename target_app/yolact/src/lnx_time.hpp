/*----------------------------------------------------------------------------
**      _____
**     *     *
**    *____   *____
**   * *===*   *==*
**  *___*===*___**  AVNET
**       *======*
**        *====*
**----------------------------------------------------------------------------
**
** This design is the property of Avnet.  Publication of this
** design is not authorized without written consent from Avnet.
**
** Disclaimer:
**    Avnet, Inc. makes no warranty for the use of this code or design.
**    This code is provided  "As Is". Avnet, Inc assumes no responsibility for
**    any errors, which may appear in this code, nor does it make a commitment
**    to update the information contained herein. Avnet, Inc specifically
**    disclaims any implied warranties of fitness for a particular purpose.
**                     Copyright(c) 2018 Avnet, Inc.
**                             All rights reserved.
**
**----------------------------------------------------------------------------
**
** Create Date:         July 16, 2018
** File Name:           lnx_time.h
**
** Tool versions:       SDSoC 2017.4
**
** Description:         Class for Linux time measurement
**
** Revision:            July 16, 2018: 1.00 Initial version
**
**----------------------------------------------------------------------------*/

#ifndef LNX_TIME_H
#define LNX_TIME_H

#include <ctime>

class lnx_timer
{
  private:    
    uint64_t lnx_clock_counter()
    {
      timespec t, c;
    
      clock_gettime(CLOCK_MONOTONIC, &t);
      clock_getres(CLOCK_MONOTONIC, &c);
    
      double   secs  = ((t.tv_sec * 1e9) + t.tv_nsec);
      uint64_t ticks = secs / (double)c.tv_nsec;
    
      return ticks;
    
    }
    
    uint64_t lnx_clock_frequency()
    {
      timespec c;
      clock_getres(CLOCK_MONOTONIC, &c);
    
      return (uint64_t) ( 1.0e9 / (double)c.tv_nsec );
    }    
  
  public:
    uint64_t cnt, tot, calls;
    lnx_timer() : cnt(0), tot(0), calls(0) {};
    inline void reset(){ tot = cnt = calls = 0; };
    inline void start(){ cnt = lnx_clock_counter(); calls++; };
    inline void stop(){ tot += (lnx_clock_counter() - cnt); };
    inline uint64_t avg() {return (tot / calls);};
    inline float    secs(){ return (float)((double)tot / (double)lnx_clock_frequency()); };
    inline float    avg_secs(){ return (float)((double)avg() / (double)lnx_clock_frequency()); };   
};

#endif

