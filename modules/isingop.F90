!###################### MODULE HANDLING EX9 ############################
module isingop
    use quantumstatesqic
    use printer
    use matrixqic
contains
!############
!SLOW VERSION
    function hampart1(N) result(pt)
        implicit none
        integer:: N, ii,jj
        complex*16, dimension(2**N, 2**N) :: pt
        complex*16, dimension(2,2,N) :: hamtmp
        pt = 0.0
        do ii = 1,N
            hamtmp = 0.0
            do jj = 1, 2
                hamtmp(jj,jj,:)%re = 1.0
                hamtmp(jj,jj,:)%im = 0.0
            end do
                hamtmp(2,2,ii)%re = -1.0
                pt = pt + znbham(hamtmp)
       end do


    end function
    function hampart2(N) result(pt)
        implicit none
        integer:: N
        complex*16, dimension(2**N, 2**N) :: pt
        integer:: ii,jj
        complex*16:: hamtmp(2,2,N),hamtmp2(2,2,N)
        pt = 0.0
        do ii = 1,N-1
            hamtmp = 0.0
            hamtmp2 = 0.0
            do jj = 1, 2
                hamtmp(jj,jj,:)%re = 1.0
                hamtmp(jj,jj,:)%im = 0.0
                hamtmp2(jj,jj,:)%re = 1.0
                hamtmp2(jj,jj,:)%im = 0.0
            end do
                hamtmp(2,2,ii)%re =0.00
                hamtmp(1,1,ii)%re =0.00
                hamtmp(1,2,ii)%re =1.00
                hamtmp(2,1,ii)%re =1.00
                hamtmp2(2,2,ii+1)%re =0.00
                hamtmp2(1,1,ii+1)%re =0.00
                hamtmp2(1,2,ii+1)%re =1.00
                hamtmp2(2,1,ii+1)%re =1.00
                !call pzm(hamtmp(:,:,1))
                pt = pt + matmul(znbham(hamtmp2),znbham(hamtmp))
       end do
   end function
  ! >END SLOW VERSION
  !#######################

!computes field term of the hamiltonian
   function hamfield(N,lambda) result(pt)
       implicit none
       integer:: N ,ii
       complex*16, dimension(2**N, 2**N) :: pt
       complex*16,dimension(2,2):: pauliz
       real*8 ::  lambda
       pauliz = 0.0
       pauliz(1,1)%re=1.0
       pauliz(1,2)%re=0.0
       pauliz(2,1)%re=0.0
       pauliz(2,2)%re=-1.0
       pt = 0.0
       do ii = 1,N
            pt = pt + getbigmat(pauliz,ii,N)
       end do
       pt = lambda*pt
  end function

!computes interaction term of the hamiltonian
  function hamint(N) result(pt)
      implicit none
      integer:: N,ii
      complex*16, dimension(2**N, 2**N) :: pt
      complex*16,dimension(2,2):: paulix

      paulix = 0.0
      paulix(1,1)%re=0.0
      paulix(1,2)%re=1.0
      paulix(2,1)%re=1.0
      paulix(2,2)%re=0.0
      pt = 0.0
      do ii = 1,N-1
            pt = pt + matmul(getbigmat(paulix,ii+1,N),getbigmat(paulix,ii,N))
     end do
     !pt = pt+matmul(getbigmat(paulix,N,N),getbigmat(paulix,1,N)) ! join end and start of the chain
 end function


!compute full hamiltonian, summing up interaction and
!field term
 function fullham(lambda,N) result(hamiltonian)
       implicit none
       integer:: N
       real*8:: lambda
       double complex, dimension(:,:) :: hamiltonian(2**N, 2**N), ham0(2**N, 2**N)
       hamiltonian = hamfield(N,lambda)+hamint(N)
end function


!compute kroeneker product of a 2x2 matrix with and Identity matrices
! I x A x I
!exploits properties of identitiy matrices.
    function getbigmat0(paulimat, index, N) result(bigmat)
        integer :: index, N, ii,jj, kk1, kk2
        integer :: helpidx(2), blocksize
        double complex , dimension(2,2) :: paulimat
        double complex, dimension(2**N,2**N) :: bigmat
        blocksize = (2**(N-index))
        bigmat = 0.0
        do ii = 1, 2
            do jj = 1, 2
                do kk1 = 0,2**(index-1)-1
                    helpidx(1)= (ii)+2*kk1
                    helpidx(2)= (jj)+2*kk1
                    do kk2 = 1, blocksize
                        bigmat((helpidx(1)-1)*blocksize+kk2 ,(helpidx(2)-1)*blocksize+kk2)  &
                            = paulimat(ii,jj)
                    end do
                end do
            end do
        end do
    end function

    !exploits properties of identitiy matrices.
    function getbigmat(inputmat, index, N) result(bigmat)
        integer :: index, N, ii,jj, kk1, kk2,idim
        integer :: helpidx(2), blocksize
        double complex , dimension(:,:) :: inputmat
        double complex, dimension(size(inputmat,dim = 1)**N,size(inputmat,dim = 1)**N) :: bigmat
        idim = size(inputmat,dim = 1)
        blocksize = (idim**(N-index))
        bigmat = 0.0
        do ii = 1, idim
            do jj = 1, idim
                do kk1 = 0,idim**(index-1)-1
                    helpidx(1)= (ii)+idim*kk1
                    helpidx(2)= (jj)+idim*kk1
                    do kk2 = 1, blocksize
                        bigmat((helpidx(1)-1)*blocksize+kk2 ,(helpidx(2)-1)*blocksize+kk2)  &
                            = inputmat(ii,jj)
                    end do
                end do
            end do
        end do
    end function
end module isingop


