module time_dependent_schrodinger
    use debugqic
    use quantumop
    use printer
    use matrixqic
    integer FFTW_FORWARD,FFTW_BACKWARD
    parameter (FFTW_FORWARD=-1,FFTW_BACKWARD=1)

    integer FFTW_REAL_TO_COMPLEX,FFTW_COMPLEX_TO_REAL
    parameter (FFTW_REAL_TO_COMPLEX=-1,FFTW_COMPLEX_TO_REAL=1)

    integer FFTW_ESTIMATE,FFTW_MEASURE
    parameter (FFTW_ESTIMATE=0,FFTW_MEASURE=1)

    integer FFTW_OUT_OF_PLACE,FFTW_IN_PLACE,FFTW_USE_WISDOM
    parameter (FFTW_OUT_OF_PLACE=0)
    parameter (FFTW_IN_PLACE=8,FFTW_USE_WISDOM=16)

    integer FFTW_THREADSAFE
    parameter (FFTW_THREADSAFE=128)
    contains

        !###############################################
        ! compute Forward and backward fourier transform
        function zfft(input, N, direction) result(output)
        !###############################################
            implicit none

            integer :: N,direction,ii
            integer*8 :: plan
            double precision :: scale = 1.0
            double complex, dimension(N), intent(INOUT):: input
            double complex, dimension(N)::output,input1

            call breakifn("FFT:  Input must be 1D double complex array", size(shape(input))  .eq. 1, .true. )
            call breakifn("FFT: Output must be 1D double complex array", size(shape(output)) .eq. 1, .true. )
            n = size(input, dim=1)
            call breakifn("FFT: Input and output must have the same dimension", size(input) .eq.size(output))


            if (direction .eq. 1) then
                call dfftw_plan_dft_1d(plan,N,input1,output,1,0)
                scale = dble(1/dble(N))
            else if (direction .eq. -1) then
                call dfftw_plan_dft_1d(plan,N,input1,output,-1,0)
            else
                call breakifn("Invalid Direction of the FFT direct +1, inverse -1", .false., .true.)
            endif
            input1 = input
            call dfftw_execute_dft(plan, input1, output)
            call dfftw_destroy_plan(plan)
            output = output*scale
        end function

        !##################################
        function init(x1,x2,L, w, h ,m) result(eigveczero)
        !##################################
        implicit none
        integer*4 :: L,II
        double complex, dimension(2*L+1,2*L+1) :: hamiltonian
        double complex, dimension(2*L+1) :: eigveczero
        double precision, dimension(2*L+1) :: eigf
        double complex :: ek = 0., harm=0.
        double precision :: x1,x2,eps,m,h,w,ko,hm

        !--------------------------------------------
        !let's evalutate the first eigenstate as done for the last exercise
            hm=h*0.5*m !set constants
            ko=m*w*w*0.5
            eps= (x2-x1)/(2*L+1)


            harm%re = ko
            ek%re = -hm


            call setenv(x1,x2,L)
            hamiltonian = +K_1d(ek)+x2_1d(harm)  !here we get the hamiltonian
            eigveczero = hamiltonian(:,1)

            !compute eigenvalues and eigenvectors
            call eigz(hamiltonian,2*L+1,eigf)
            !get the first eigenvector
            eigveczero = hamiltonian(:,1)

        end function init
        !###############################################
        !update temoral steps
        function update(vec,vel,x1,x2,L,w,tzero,dt,m,h) result(vec1)
        !###############################################
        implicit none
        double complex, dimension(2*l+1):: vec1, vec,vec2,vechelp
        double precision :: vel,dt,w,m,k,x1,x2,tzero,h
        double precision :: PI=3.14159265358, value, helper1, helper2
        integer ::ii,L,N
        double precision :: x, eps
        integer*8:: plan

            eps = (x2-x1)/dble(2*L+1)
            vec1=vec
            !-----------Potential --------------
            do ii = 1, 2*l+1
                x = x1+ dble(ii-1)* eps
                !helper1 = ((x-vel*(tzero+0.5*dt))**2)*dt
                helper1 = dt*(x**2)-vel*x*dt*(2*tzero+dt*dt)+(vel**2)*(tzero+dt)*dt*tzero
                vec1(ii) = vec1(ii)*exp(-0.25*m*w*w*(1/h)*dcmplx(0.D0, helper1 ))
            end do
            !---------- Fourier transform ----------
            vec2 = zfft(vec1,size(vec1),1)
            N= 2*l+1
            !---------- Kinetic ----------
            do ii = 1, l
                k=((2*PI*dble((ii-1))/(x2-x1)))
                vec2(ii) = vec2(ii)*exp(dcmplx(0.D0,-(0.5/m)*h*k*k*dt))
             end do
             do ii = l+1, 2*l+1
                 k=2*PI*dble((ii-2*l-2))/((x2-x1))
                 vec2(ii) = vec2(ii)*exp(dcmplx(0.D0,-(0.5/m)*h*k*k*dt))
             end do
             !---------- Fourier inverse transform ----------
             vec1 = zfft(vec2,size(vec1),-1)
             !------  Potential part ----------
             do ii = 1, 2*l+1
                 x = x1+ dble(ii-1)* eps
                 !helper1 = ((x-vel*(tzero+0.5*dt))**2)*dt
                 helper1 = dt*(x**2)-vel*x*dt*(2*tzero+dt*dt)+(vel**2)*(tzero+dt)*dt*tzero
                 vec1(ii) = vec1(ii)*exp(-0.25*m*w*w**(1/h)*cmplx(0.D0, helper1 ))
             end do
             vec1= vec1/sqrt(norm2(vec1%re)**2+norm2(vec1%im)**2)

        end function update

end module time_dependent_schrodinger
