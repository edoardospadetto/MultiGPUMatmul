module quantumstatesqic
    !mixed wave function handler
    use debugqic
    use matrixqic
    type compositewf
        integer ::number
        integer ::states !of the single object ex. spin |0> |1> hence 2
        integer :: nclassicalmixture
        double precision, dimension(:), allocatable :: probs
        double complex, dimension(:,:), allocatable :: wave
        contains
            final :: destruct
    end type

    type dnsmat
        integer ::nobodies
        integer :: nostatesxb !of the single object ex. spin |0> |1> hence 2
        double complex, dimension(:,:), allocatable :: mat
        contains
            final :: destroy
    end type

    interface compositewf
    module procedure :: init
    end interface

    interface dnsmat
        module procedure :: createdm
        end interface

    contains
        !Initializers and Deconstructors
        subroutine destruct(instance)
            type(compositewf) :: instance
            write(*,*) "destroying object"
            deallocate(instance%probs)
            deallocate(instance%wave)
        end subroutine destruct
        subroutine destroy(instance)
            type(dnsmat)  :: instance
            write(*,*) "destroying object"
            deallocate(instance%mat)
        end subroutine destroy
        !Init manybody type
        !#####################################################
        function init(numberofbodies, states, cases) RESULT(system)
        !#####################################################
            implicit none
            type(compositewf) :: system
            integer :: numberofbodies, states,cases
            system%number = numberofbodies
            system%states = states
            allocate(system%probs(cases))
            call breakifn("probs not allocated" , allocated(system%probs), .true. )
            allocate(system%wave(states**numberofbodies,cases))
            call breakifn("wave not allocated" , allocated(system%wave), .true. )
        end function

        !Generate Separable States, set of vector, rows are components of hilber basis
        !columns are the different bodies
        !#########################################################
        function newsystemsep(nostate, nbody) result(thewavefunction)
            implicit none
            double precision :: thenorm
            integer :: nostate, nbody, ii
            double complex , dimension(nostate,nbody) ::thewavefunction
            thewavefunction = rgzm(nostate, nbody)

            do ii = 1, nbody
                thenorm =  sqrt(norm2(thewavefunction(:,ii)%re)**2+norm2(thewavefunction(:,ii)%im)**2)
                thewavefunction(:,ii)= thewavefunction(:,ii)/ thenorm
            end do
        end function

        !Generate random wavefunction, of a non separable subsystem of n bodies
        !#########################################################
        function newsystemnonseparable(nostate, nbody) result(thewavefunction)
        !#########################################################
            implicit none
            double precision :: thenorm
            integer :: nostate, nbody, ii
            double complex , dimension(nostate**nbody,1) ::thewavefunction
            thewavefunction = rgzm(nostate**nbody,1)
            thenorm =  sqrt(norm2(thewavefunction(:,1)%re)**2+norm2(thewavefunction(:,1)%im)**2)
            thewavefunction= thewavefunction/ thenorm
        end function


        !init dnsmat type
        !#####################################################
        function createdm(numberofbodies, states,  themat) RESULT(dmat)
        !#####################################################
            implicit none
            type(dnsmat) :: dmat
            integer :: numberofbodies, states
            double complex, dimension(:,:) :: themat
            if (numberofbodies*states .ne. 0) then
                call breakifn("Wrong row number createdm", states**numberofbodies .eq. size(themat, dim= 1), .true.)
                call breakifn("Wrong cols number createdm", states**numberofbodies .eq. size(themat, dim= 2), .true.)
            else
                print*, "Warning CREATED: number of bodies or n.o. states not specified, checks skipped"
            end if
            dmat%nobodies = numberofbodies
            dmat%nostatesxb = states
            allocate(dmat%mat(size(themat, dim= 1),size(themat, dim= 2)))
            call breakifn("matr not allocated" , allocated(dmat%mat), .true. )
            dmat%mat = themat
        end function

        !#####################################################
        function MergeWF(states) result(thiswave)
        !Tensor product of thiswave functions
        !#####################################################
        implicit none
        double complex, dimension(:,:) :: states
        type(compositewf) :: thiswave
        integer:: counter(size(states, dim= 2)), nbodies, nstates,ii,jj,kk
        counter = 1
        nbodies = size(states, dim= 2)
        nstates = size(states, dim= 1)
        thiswave = init(nbodies, nstates, 1)
        thiswave%wave = 1.0
        thiswave%probs(1) = 1.0
        jj=1
        !The result index runs with jj, we keep track of the states of the various particle
        !with that counter, that simply counts in the basis of the number of states
        !for example , if there are 2 states it count in binary, if 5 in base 5.

        !notice, the first column of the input will be the lowest order element in the RESULT
        !example : psi = (|00> |01> |10> |11>) the el on the right is the first column of the input
         do
             !write(*,*) counter, jj !check if there's corresdimensionpondence between j and the counter
             !do stuff
             !JUST COMPUTING TENSOR PRODUCT COEFFICIENTS
             !--------------------------------
             do kk = 1 , nbodies
                !print*, counter(kk), kk
                thiswave%wave(jj,1) = thiswave%wave(jj,1)*states(counter(kk),kk)
             end do
             !-------------------------------
            !CHECK IF WE ARE DONE WITH THE INDICES
            !was it the last iteration?
            if (product(counter)  .eq. nstates**nbodies) then
                !print*, counter
                exit
            end if
            !no, so move by one
            counter(1) = counter(1)+1
            jj= jj+1
            !Check for rests in the sum.
            do ii =1,nbodies
                if (counter(ii) .gt. nstates) then
                    counter(ii) = 1
                    counter(ii+1) = counter(ii+1)+1
                end if
            end do
        end do
        print*, "done!"
        end function



    !Recover density matrix from wavefunction, both pure or mixed
    !Here we start from matrix in which columns represents variuous mixture of a wave function
    !while rows the states coefficient
    !Also an array with the probabilities of the mixtures must be provided
    !The output is a dns type matrix, but incomplete, number of states and bodies is not provided in the
    !type, better to use the following function that works withg the user defined type and sets automaticcaly
    !these information in the dnsmat type

    !#####################################################
    function BDM(states,probs) result(dns_result)
    !#####################################################
        use debugqic
        implicit none

        integer :: ii, dim
        double precision ,dimension(:) :: probs
        double complex, dimension(:,:) :: states
        double complex, dimension(size(states,dim=1),size(states,dim=1)) :: density_matrix, tmp
        type(dnsmat) dns_result
        density_matrix =0.0

        call breakifn("ncol must be equal to number of states", (size(states,dim=2) .eq. size(probs)), .true. )
        dim =size(states,dim=1)
        do ii =1,size(probs)
            tmp = 0.0
            tmp =   probs(ii)*matmul(RESHAPE(states(:,ii),(/dim, 1/)), RESHAPE(conjg(states(:,ii)),(/1,dim/)))
            density_matrix = density_matrix +tmp
        end do

        dns_result =  createdm(0,0,density_matrix)

    end function

    !As function above but works with the user defined type many body
    !#####################################################
    function BDMmb(mbwave) result(dns_result)
    !#####################################################
        use debugqic
        implicit none

        integer :: ii, dim
        type(compositewf) :: mbwave
        double complex, dimension(:,:), allocatable :: density_matrix
        type(dnsmat) :: dns_result
        !print*, mbwave%number,mbwave%states
        allocate(density_matrix(size(mbwave%wave,dim=1),size(mbwave%wave,dim=1)))
        call breakifn("not allocated" , allocated(density_matrix), .true.)
        density_matrix = 0.0
        !print *, "hello"
        call breakifn("ncol must be equal to number of mixtures", (size(mbwave%wave,dim=2) .eq. size(mbwave%probs)), .true. )
        dim =size(mbwave%wave,dim=1)
        density_matrix = 0.0
        do ii =1,size(mbwave%probs)
            density_matrix= density_matrix+   mbwave%probs(ii)* &
                matmul(RESHAPE(mbwave%wave(:,ii),(/dim, 1/)), RESHAPE(conjg(mbwave%wave(:,ii)),(/1,dim/)))
        end do

        dns_result =  createdm(mbwave%number,mbwave%states,density_matrix)


    end function

    !Computes reduce density matrix, from the dnsmat object, and an array
    !The array defines which subsystem to keep
    !Works for generic number of bodies and generic number of states.
    !It's slow check the next one
    !#####################################################
    function marginalizeDMMB(fulldm, bodiestokeep) result(dns_result)
    !#####################################################
        use printer
        use matrixqic
        type(dnsmat) :: fulldm, dns_result
        integer, dimension(:) :: bodiestokeep
        integer, dimension(fulldm%nobodies - size(bodiestokeep, dim=1)):: bodiestodump
        integer :: ii,jj,kk, counter(fulldm%nobodies,2), maxcnt(fulldm%nobodies), helpi, helpj
        double complex , dimension(fulldm%nostatesxb**size(bodiestokeep),fulldm%nostatesxb**size(bodiestokeep)) :: tmpres
        jj=1
        !write(*,*) "ok? "
        do ii = 1 , fulldm%nobodies
             if (any(bodiestokeep == ii))  then
                 continue
            else
                bodiestodump(jj) = ii
                jj = jj+1
            end if
        end do
        !print*, "ahoo"
        call breakifn("Something wrong in what to keep", jj-1 .eq. size(bodiestodump, dim= 1), .true. )
        tmpres = 0.0
        call breakifn("Error marginalizeDMMB : Number of bodies not specified", fulldm%nobodies .gt. 0, .true. )
        call breakifn("Error marginalizeDMMB : Number of states per body not specified", fulldm%nostatesxb .gt. 0, .true. )
        do ii = 1, size(fulldm%mat, dim=1)
            do jj = 1, size(fulldm%mat, dim = 1)
                !I need to convert the indices in identifier for the single particles
                helpi=ii-1
                helpj=jj-1
                do kk = 1,fulldm%nobodies
                    counter(kk,1) = helpi - int(fulldm%nostatesxb)*(helpi / int(fulldm%nostatesxb))
                    counter(kk,2) = helpj - int(fulldm%nostatesxb)*(helpj / int(fulldm%nostatesxb))
                    helpi=helpi / int(fulldm%nostatesxb)
                    helpj=helpj / int(fulldm%nostatesxb)
                    !print*, helpj
                end do
                !Now i have to select the indices that i want to keep
                !Now helpi/j is used for another purpose.
                helpi=1
                helpj=1
                !write(*,*), counter(:,1), counter(:,2)
                do kk= 1, size(bodiestokeep,1)
                    helpi=helpi+counter(bodiestokeep(kk),1)*(fulldm%nostatesxb)**(kk-1)
                    helpj=helpj+counter(bodiestokeep(kk),2)*(fulldm%nostatesxb)**(kk-1)
                end do

                if (all(counter(bodiestodump, 1) .eq. counter(bodiestodump, 2))) then
                    !write (*,*) ii , jj , helpi, helpj
                    !write(*,*) bodiestodump, bodiestokeep
                    !write(*,*) counter(bodiestodump,1), counter(bodiestokeep,1), helpi
                    tmpres(helpi,helpj) = tmpres(helpi,helpj) +fulldm%mat(ii,jj)
                end if
            end do
        end do
        dns_result = createdm(size(bodiestokeep), fulldm%nostatesxb,tmpres)
    end function

    !Computes reduce density matrix, from the dnsmat object, and an array
    !The array defines which subsystem to keep
    !Works for generic number of bodies and generic number of states.
    !This is the good one
    !#####################################################
    function marginalizeDMMB2(fulldm, bodiestokeep) result(dns_result)
    !#####################################################
        use printer
        use matrixqic
        type(dnsmat) :: fulldm, dns_result
        integer, dimension(:) :: bodiestokeep
        integer, dimension(fulldm%nobodies - size(bodiestokeep, dim=1)):: bodiestodump
        integer :: ii,jj,kk,mm,nn,tt, counter(fulldm%nobodies,2), helpk, helpj,helpi
        double complex , dimension(fulldm%nostatesxb**size(bodiestokeep),fulldm%nostatesxb**size(bodiestokeep)) :: tmpres
        jj=1
        do ii = 1 , fulldm%nobodies
             if (any(bodiestokeep == ii))  then
                 continue
            else
                bodiestodump(jj) = ii
                jj = jj+1
            end if
        end do

        call breakifn("Something wrong in what to keep", jj-1 .eq. size(bodiestodump, dim= 1), .true. )
        tmpres = 0.0
        call breakifn("Error marginalizeDMMB : Number of bodies not specified", fulldm%nobodies .gt. 0, .true. )
        call breakifn("Error marginalizeDMMB : Number of states per body not specified", fulldm%nostatesxb .gt. 0, .true. )
        do ii = 1, fulldm%nostatesxb**size(bodiestokeep,1)
            do jj = 1, fulldm%nostatesxb**size(bodiestokeep,1)
                !print*, " "
                helpi = ii-1
                helpj = jj-1
                do tt= 1, size(bodiestokeep, dim=1)
                    counter(bodiestokeep(tt),1)= helpi - (helpi/fulldm%nostatesxb)*fulldm%nostatesxb
                    counter(bodiestokeep(tt),2)= helpj - (helpj/fulldm%nostatesxb)*fulldm%nostatesxb
                    helpi =  (helpi/fulldm%nostatesxb)
                    helpj =  (helpj/fulldm%nostatesxb)
                end do
                do kk = 1,fulldm%nostatesxb**size(bodiestodump,1)
                    helpk = kk -1
                    do tt = 1, size(bodiestodump, dim=1)
                        counter(bodiestodump(tt),1)= helpk - (helpk/fulldm%nostatesxb)*fulldm%nostatesxb
                        counter(bodiestodump(tt),2)= helpk - (helpk/fulldm%nostatesxb)*fulldm%nostatesxb
                        helpk =  (helpk/fulldm%nostatesxb)
                    end do

                    mm =1
                    nn =1

                    do tt= 1, size(counter, dim=1)
                        mm = mm + counter(tt,2)* fulldm%nostatesxb**(tt-1)
                        nn = nn + counter(tt,1)* fulldm%nostatesxb**(tt-1)
                    end do
                        tmpres(ii,jj) = tmpres(ii,jj) + fulldm%mat(nn,mm)
                end do
            end do
        end do
        dns_result = createdm(size(bodiestokeep), fulldm%nostatesxb,tmpres)
    end function

    function znbham(manyhamiltonian) result(tmp2)
        implicit none
        double complex, dimension(:,:,:):: manyhamiltonian
        double complex, dimension(:,:), allocatable:: tmp0, tmp1,tmp2
        integer:: ii, sizetmp(2)
        allocate(tmp1(size(manyhamiltonian,dim=1),size(manyhamiltonian,dim=2)))
        tmp1 = manyhamiltonian(:,:,1)
        do ii = 2, size(manyhamiltonian, dim = 3)
            if (.not. allocated(tmp1)) then
                allocate(tmp1(size(tmp2,dim=1),size(tmp2, dim=2)))
                tmp1 = tmp2
            end if
            if(allocated(tmp2)) then
                deallocate(tmp2)
            end if
            sizetmp(1)=size(tmp1, dim = 1)*size(manyhamiltonian, dim = 1)
            sizetmp(2)=size(tmp1, dim = 2)*size(manyhamiltonian, dim = 2)
            allocate(tmp2(sizetmp(1), sizetmp(2)))
            tmp2 = zkrcprod(manyhamiltonian(:,:,ii), tmp1)
            deallocate(tmp1)
        end do
        !print*, size(manyhamiltonian,dim = 1)**size(manyhamiltonian,dim = 3)
        !print*, size(tmp2, dim=1)
        call breakifn("output not correct" , &
            size(tmp2, dim=1) .eq. (size(manyhamiltonian,dim = 1)**size(manyhamiltonian,dim = 3)),&
             .true.)



    end function


end module quantumstatesqic
