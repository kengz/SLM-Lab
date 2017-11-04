import pytest


class TestData:
    '''
    Base class for unit testing data provided to a network
    '''

    @pytest.mark.parametrize("batch", [None, None, None])
    def test_zero(self, batch):
        '''
        Checks that a batch of data is not zero
        batch: tensor representing a batch of data. May be of arbitrary
        dimension, but first dimension always represents batch size
        returns: true if batch is not all zero, false otherwise'''
        # TODO: implement basic version
        assert batch == None

    @pytest.mark.parametrize("dataloader", [None, None, None])
    def test_different(self, dataloader):
        '''
        Checks that batches of data are changing
        returns: true if two batches are not equal to each other,
                 false otherwise
        '''
        # TODO: decide on format for dataloader and implement basic version
        assert dataloader == None
