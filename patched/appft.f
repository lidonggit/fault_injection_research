         subroutine appft (niter, total_time, verified)
         implicit none
         include  'global.h'
         integer niter
         double precision total_time
         logical verified
!
! Local variables
!
         integer i, j, k, kt, n12, n22, n32, ii, jj, kk, ii2, ik2
         double precision ap
         double precision twiddle(nx+1,ny,nz)

         double complex xnt(nx+1,ny,nz),y(nx+1,ny,nz),
     &                  pad1(128),pad2(128)
         common /mainarrays/ xnt,pad1,y,pad2,twiddle

         double complex exp1(nx), exp2(ny), exp3(nz)

c---------------------------------------------------------------------
         intrinsic signal
         external launch_fi_thread
         external sig_handler
c---------------------------------------------------------------------

c---------------------------------------------------------------------
         character(10) :: iter_string, time_string, tr_string, sr_string
         integer memsize
         integer numarg
         integer iter_num
         integer mem_size
         double precision exec_t, t_rand, s_rand

         numarg = iargc ( )

         if (numarg .ne. 4) then
           WRITE (*,*) 'mg [iter] [exec_time] [time_rand] [space_rand]'
           STOP
         else
           call getarg ( 1, iter_string )
           call getarg ( 2, time_string )
           call getarg ( 3, tr_string )
           call getarg ( 4, sr_string )
         endif

         read (iter_string, *) iter_num
         read (time_string, *) exec_t
         read (tr_string, *) t_rand
         read (sr_string, *) s_rand
c---------------------------------------------------------------------


         do i=1,15
           call timer_clear(i)
         end do

         call timer_start(2)
         call compute_initial_conditions(xnt,nx,ny,nz)

         call CompExp( nx, exp1 )
         call CompExp( ny, exp2 )
         call CompExp( nz, exp3 )
         call fftXYZ(1,xnt,y,exp1,exp2,exp3,nx,ny,nz)
         call timer_stop(2)

         if (timers_enabled) call timer_start(13)

         n12 = nx/2
         n22 = ny/2
         n32 = nz/2
         ap = - 4.d0 * alpha * pi ** 2
         do i = 1, nz
           ii = i-1-((i-1)/n32)*nz
           ii2 = ii*ii
           do k = 1, ny
             kk = k-1-((k-1)/n22)*ny
             ik2 = ii2 + kk*kk
             do j = 1, nx
                 jj = j-1-((j-1)/n12)*nx
                 twiddle(j,k,i) = exp(ap*dble(jj*jj + ik2))
               end do
            end do
         end do
         if (timers_enabled) call timer_stop(13)

         if (timers_enabled) call timer_start(12)
         call compute_initial_conditions(xnt,nx,ny,nz)
         if (timers_enabled) call timer_stop(12)
         if (timers_enabled) call timer_start(15)
         call fftXYZ(1,xnt,y,exp1,exp2,exp3,nx,ny,nz)
         if (timers_enabled) call timer_stop(15)

c-----------------------------------------------------------------------------
      memsize = (nx+1)*ny*nz
c     WRITE (*,*) memsize
      if (iter_num .ne. 0) then
      call signal(5, sig_handler)
      call launch_fi_thread(xnt,memsize,iter_num,exec_t,t_rand,s_rand)
      endif
c-----------------------------------------------------------------------------

         call timer_start(1)

         do kt = 1, niter
           if (timers_enabled) call timer_start(11)
           call evolve(xnt,y,twiddle,nx,ny,nz)
           if (timers_enabled) call timer_stop(11)
           if (timers_enabled) call timer_start(15)
           call fftXYZ(-1,xnt,xnt,exp1,exp2,exp3,nx,ny,nz)
           if (timers_enabled) call timer_stop(15)
           if (timers_enabled) call timer_start(10)
           call CalculateChecksum(sums(kt),kt,xnt,nx,ny,nz)
           if (timers_enabled) call timer_stop(10)
         end do

         call timer_stop(1)
!
! Verification test.
!
         if (timers_enabled) call timer_start(14)
         call verify(nx, ny, nz, niter, sums, verified)
         if (timers_enabled) call timer_stop(14)


         total_time = timer_read(1)
c         if (.not.timers_enabled) return

c         print*,'FT subroutine timers '
c         write(*,40) 'FT total                  ', timer_read(1)
c         write(*,40) 'WarmUp time               ', timer_read(2)
c         write(*,40) 'fftXYZ body               ', timer_read(3)
c         write(*,40) 'Swarztrauber              ', timer_read(4)
c         write(*,40) 'X time                    ', timer_read(7)
c         write(*,40) 'Y time                    ', timer_read(8)
c         write(*,40) 'Z time                    ', timer_read(9)
c         write(*,40) 'CalculateChecksum         ', timer_read(10)
c         write(*,40) 'evolve                    ', timer_read(11)
c         write(*,40) 'compute_initial_conditions', timer_read(12)
c         write(*,40) 'twiddle                   ', timer_read(13)
c         write(*,40) 'verify                    ', timer_read(14)
c         write(*,40) 'fftXYZ                    ', timer_read(15)
         write(*,40) 'Time in seconds            ', total_time
   40    format(A15,' =',F9.4)

         return
      end
