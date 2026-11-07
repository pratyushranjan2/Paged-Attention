class BlockAllocator:
    
    def __init__(self, num_blocks):
        self.free_blocks = list(range(num_blocks))

    def allocate(self):
        return self.free_blocks.pop()

    def release(self, block_id):
        return self.free_blocks.append(block_id)