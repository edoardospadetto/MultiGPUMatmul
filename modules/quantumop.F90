module quantumop
use debugqic
use printer
    !Buided for exercise 6, helper module compute eigenstate quantum harmonic oscilltor
    double precision :: xmin,xmax,e
    integer:: dim

    contains
        ! set variables of the simulation
        subroutine setenv(x1,x2,L)
            implicit none
            double precision :: x1,x2
            integer :: L
            xmin = x1
            xmax = x2
            dim  = L
            e = (x2-x1)/real(2*L+1)
        end subroutine setenv
        !###############################################
        ! compute kinetic energy therm
        function K_1d(cnst) result(K)
        !###############################################
            implicit none
            integer :: N,ii

            double  complex :: cnst
            double  complex :: K(2*dim+1,2*dim+1)
            K= (.0,.0)

            do ii = 2, 2*dim
                    K(ii,ii) =(-2.0,.0)
                    K(ii+1,ii)=(1.0,.0)
                    K(ii-1,ii)=(1.0,.0)
            end do
            K(1,1) =(-2.0,.0)
            K(2*dim+1,2*dim+1) =(-2.0,.0)
            K(2,1)=(1.0,.0)
            K(2*dim,2*dim+1)=(1.0,.0)

            K=cnst*K*(1/e**2)
        end function K_1d
        !###############################################
        !compute the potential energy therm
        function x2_1d(cnst) result(x2)
        !###############################################
            implicit none
            integer :: ii
            double complex :: x2(2*dim+1,2*dim+1), cnst
            double complex :: tmp
            x2 = (0.0,0.0)
            do ii = 1, 2*dim+1
                tmp%im=0.0
                tmp%re = e*(ii)+xmin
                x2(ii,ii) = tmp**2
            end do
            x2=cnst*x2
        end function x2_1d
        !########################################################
    

end module
